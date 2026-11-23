from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor

from gravity_gnn.data import (
    GraphData,
    GroupSplit,
    load_graph_npz,
    strict_partition_edge_mask,
)
from gravity_gnn.models import HeterogeneousFacilityGravityGNN
from gravity_gnn.training import (
    fit_model,
    fit_model_fixed_epochs,
    seed_everything,
    standardize_building_features,
)


DEFAULT_STUDIES = ("greater_london", "greater_manchester")
DEFAULT_RELATION_ACCESS_SHIFTS = (0.0, 5.0, 10.0, 15.0, 20.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the relation-preserving facility-to-LSOA Gravity-GNN"
    )
    parser.add_argument(
        "--case-root", type=Path, default=Path("data/processed/case_studies")
    )
    parser.add_argument(
        "--study",
        action="append",
        choices=DEFAULT_STUDIES,
        help="Target study; repeat for both. Defaults to both case studies.",
    )
    parser.add_argument(
        "--target",
        choices=("ahah", "domain_h", "domain_g", "domain_e", "domain_r"),
        default="ahah",
        help=(
            "LSOA prediction target. Non-AHAH targets are aligned from lsoas.parquet "
            "without changing the graph, context features, or spatial folds."
        ),
    )
    parser.add_argument("--seeds", default="20260827")
    parser.add_argument(
        "--hgb-seed",
        type=int,
        default=20260827,
        help=(
            "Fixed context-HGB seed shared across neural seeds. The resulting "
            "strict-fold residual problems are cached within each city."
        ),
    )
    parser.add_argument("--test-fold", type=int, action="append")
    parser.add_argument("--epochs", type=int, default=240)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--message-dim", type=int, default=16)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--area-backbone", choices=("mlp", "spline"), default="mlp")
    parser.add_argument(
        "--outer-refit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Refit on train plus validation for the selected number of epochs.",
    )
    parser.add_argument(
        "--hgb-residual",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fit the GNN to strictly cross-fitted context-HGB residuals.",
    )
    parser.add_argument(
        "--residual-context-covariates",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Let the residual GNN re-use the 103 context covariates.",
    )
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument(
        "--kernel-mode",
        choices=("learned", "fixed", "uniform", "gated"),
        default="learned",
    )
    parser.add_argument(
        "--edge-basis",
        choices=("full", "mass_only"),
        default="full",
        help=(
            "Use full to expose routed time to both the edge network and gravity "
            "kernel. Use mass_only to make the gravity kernel the only facility-edge "
            "distance channel."
        ),
    )
    parser.add_argument(
        "--queen-message-passing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--cross-city-pooling",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use all labelled LSOAs in the other city as auxiliary training data.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/heterogeneous_facility_gravity_gnn.json"),
    )
    parser.add_argument(
        "--prediction-dir",
        type=Path,
        default=Path("results/heterogeneous_facility_predictions"),
    )
    parser.add_argument(
        "--export-relation-access",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Export strict-OOF HGGE relation-access contrasts obtained by "
            "comparing observed routed times with a 40-minute reference and "
            "with within-relation walking-time shifts."
        ),
    )
    parser.add_argument(
        "--relation-access-shifts",
        default=",".join(str(value) for value in DEFAULT_RELATION_ACCESS_SHIFTS),
        help="Comma-separated walking-time shifts for the learned relation exposure.",
    )
    parser.add_argument(
        "--exclude-relations",
        default="",
        help=(
            "Comma-separated relation names whose edges are removed before every "
            "training and evaluation split. Intended for pre-specified facility-"
            "content ablations; isolated facility nodes are retained."
        ),
    )
    return parser


def metrics(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    residual = prediction - y_true
    total = np.sum((y_true - y_true.mean()) ** 2)
    return {
        "r2": float(1.0 - np.sum(residual**2) / max(float(total), 1e-12)),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
    }


def parse_relation_access_shifts(value: str) -> tuple[float, ...]:
    shifts = tuple(float(item) for item in value.split(",") if item.strip())
    if not shifts or shifts[0] != 0.0:
        raise ValueError("relation-access shifts must start with 0")
    if any(shift < 0.0 or shift > 40.0 for shift in shifts):
        raise ValueError("relation-access shifts must lie between 0 and 40 minutes")
    if tuple(sorted(set(shifts))) != shifts:
        raise ValueError("relation-access shifts must be unique and increasing")
    return shifts


def exclude_relation_edges(
    data: GraphData,
    relation_names: list[str],
    excluded_relations: tuple[str, ...],
) -> GraphData:
    """Remove named relation edges without changing nodes or relation indices."""

    if not excluded_relations:
        return data
    unknown = sorted(set(excluded_relations) - set(relation_names))
    if unknown:
        raise ValueError(f"unknown excluded relations: {', '.join(unknown)}")
    edge_mask = torch.ones(data.num_edges, dtype=torch.bool)
    for relation in excluded_relations:
        edge_mask &= data.edge_type != relation_names.index(relation)
    if not edge_mask.any():
        raise ValueError("relation exclusion removed every graph edge")
    return data.with_edge_mask(edge_mask)


def prepare_fitted_graph_for_inference(
    data: GraphData,
    train_mask: torch.Tensor,
    device: str | torch.device,
) -> tuple[GraphData, torch.Tensor, torch.Tensor]:
    """Recreate the exact feature and target scaling used during model fitting."""

    device = torch.device(device)
    prepared = data.to(device)
    train_mask = train_mask.to(device)
    standardized_x, _, _ = standardize_building_features(prepared, train_mask)
    prepared = replace(prepared, x=standardized_x)
    train_values = prepared.y[train_mask]
    if prepared.sample_weight is None:
        target_mean = train_values.mean()
        target_variance = (train_values - target_mean).square().mean()
    else:
        weights = prepared.sample_weight[train_mask]
        weight_sum = weights.sum().clamp_min(1e-8)
        target_mean = torch.sum(weights * train_values) / weight_sum
        target_variance = (
            torch.sum(weights * (train_values - target_mean).square()) / weight_sum
        )
    target_scale = torch.sqrt(target_variance).clamp_min(1e-6)
    return prepared, target_mean, target_scale


def relation_access_counterfactuals(
    model: HeterogeneousFacilityGravityGNN,
    data: GraphData,
    train_mask: torch.Tensor,
    target_node_limit: int,
    target_labelled_nodes: np.ndarray,
    relation_count: int,
    shifts: tuple[float, ...],
    residual_weight: float,
    device: str | torch.device,
    reference_minutes: float = 40.0,
) -> np.ndarray:
    """Return learned access benefits for every target LSOA and relation.

    For relation ``r`` and shift ``d``, the contrast is the fitted GNN prediction
    when all retained ``r`` edges are placed at the 40-minute reference minus the
    prediction when their observed times are increased by ``d`` and clipped at
    40 minutes. The fitted outer-fold model, all other relations, context, and
    Queen topology remain fixed. Positive values therefore mean that the learned
    AHAH surface assigns a benefit to the observed/shifted access relative to the
    weak-access reference. Census health never enters this computation.
    """

    prepared, target_mean, target_scale = prepare_fitted_graph_for_inference(
        data, train_mask, device
    )
    model.eval()
    labelled = torch.as_tensor(
        target_labelled_nodes, dtype=torch.long, device=prepared.x.device
    )
    destination = prepared.edge_index[1]
    original_impedance = prepared.edge_impedance
    output = np.empty(
        (relation_count, len(target_labelled_nodes), len(shifts)), dtype=np.float32
    )

    def predict(edge_impedance: torch.Tensor) -> torch.Tensor:
        graph = replace(prepared, edge_impedance=edge_impedance)
        return model(graph) * target_scale + target_mean

    with torch.no_grad():
        observed_prediction = predict(original_impedance)
        for relation in range(relation_count):
            selected = (prepared.edge_type == relation) & (
                destination < target_node_limit
            )
            if not torch.any(selected):
                raise RuntimeError(f"target graph has no edges for relation {relation}")
            reference_impedance = original_impedance.clone()
            reference_impedance[selected] = reference_minutes
            reference_prediction = predict(reference_impedance)
            for shift_index, shift in enumerate(shifts):
                if shift == 0.0:
                    shifted_prediction = observed_prediction
                else:
                    shifted_impedance = original_impedance.clone()
                    shifted_impedance[selected] = torch.clamp(
                        original_impedance[selected] + shift,
                        max=reference_minutes,
                    )
                    shifted_prediction = predict(shifted_impedance)
                learned_access = residual_weight * (
                    reference_prediction - shifted_prediction
                )
                output[relation, :, shift_index] = (
                    learned_access[labelled].detach().cpu().numpy()
                )
    return output


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_study(
    case_root: Path, study: str, target_name: str = "ahah"
) -> tuple[GraphData, list[str], np.ndarray, np.ndarray]:
    case_dir = case_root / study
    graph_path = case_dir / "facility_lsoa_graph.npz"
    data = load_graph_npz(graph_path)
    payload = np.load(graph_path, allow_pickle=False)
    relation_names = payload["relation_names"].astype(str).tolist()
    lsoa_codes = payload["lsoa_codes"].astype(str)
    fold_plan = json.loads(
        (case_dir / "spatial_folds_5.json").read_text(encoding="utf-8")
    )
    folds = np.asarray(
        [fold_plan["lsoa_to_fold"][code] for code in lsoa_codes], dtype=int
    )
    labelled_nodes = np.flatnonzero(data.building_mask.numpy())
    if len(labelled_nodes) != len(lsoa_codes):
        raise ValueError(f"LSOA code count differs from labelled nodes for {study}")
    if target_name != "ahah":
        target_frame = pd.read_parquet(case_dir / "lsoas.parquet")
        if target_name not in target_frame.columns:
            raise ValueError(f"missing target {target_name} for {study}")
        aligned = (
            target_frame[["lsoa21cd", target_name]]
            .set_index("lsoa21cd")
            .reindex(lsoa_codes)[target_name]
            .to_numpy(dtype=float)
            .copy()
        )
        if not np.isfinite(aligned).all():
            raise ValueError(f"target {target_name} has missing values for {study}")
        target = data.y.clone()
        target[labelled_nodes] = torch.as_tensor(aligned, dtype=target.dtype)
        data = replace(data, y=target)
    if data.feature_names is None:
        raise ValueError(f"missing feature names for {study}")
    return data, relation_names, lsoa_codes, folds


def target_fold_split(
    data: GraphData, folds: np.ndarray, test_fold: int
) -> tuple[GraphData, GroupSplit, int]:
    unique_folds = sorted(int(value) for value in np.unique(folds))
    if test_fold not in unique_folds:
        raise ValueError(f"test fold {test_fold} is not in the fold plan")
    position = unique_folds.index(test_fold)
    validation_fold = unique_folds[(position + 1) % len(unique_folds)]
    labelled_nodes = np.flatnonzero(data.building_mask.numpy())
    train = torch.zeros(data.num_nodes, dtype=torch.bool)
    validation = torch.zeros(data.num_nodes, dtype=torch.bool)
    test = torch.zeros(data.num_nodes, dtype=torch.bool)
    train[labelled_nodes] = torch.as_tensor(
        (folds != test_fold) & (folds != validation_fold)
    )
    validation[labelled_nodes] = torch.as_tensor(folds == validation_fold)
    test[labelled_nodes] = torch.as_tensor(folds == test_fold)
    partition = torch.full((data.num_nodes,), -1, dtype=torch.long)
    partition[train] = 0
    partition[validation] = 1
    partition[test] = 2
    edge_mask = strict_partition_edge_mask(data, partition)
    return (
        data.with_edge_mask(edge_mask),
        GroupSplit(train=train, validation=validation, test=test, partition=partition),
        validation_fold,
    )


def target_outer_split(
    data: GraphData, folds: np.ndarray, test_fold: int
) -> tuple[GraphData, GroupSplit]:
    labelled_nodes = np.flatnonzero(data.building_mask.numpy())
    train = torch.zeros(data.num_nodes, dtype=torch.bool)
    validation = torch.zeros(data.num_nodes, dtype=torch.bool)
    test = torch.zeros(data.num_nodes, dtype=torch.bool)
    train[labelled_nodes] = torch.as_tensor(folds != test_fold)
    test[labelled_nodes] = torch.as_tensor(folds == test_fold)
    partition = torch.full((data.num_nodes,), -1, dtype=torch.long)
    partition[train] = 0
    partition[test] = 2
    edge_mask = strict_partition_edge_mask(data, partition)
    return (
        data.with_edge_mask(edge_mask),
        GroupSplit(train=train, validation=validation, test=test, partition=partition),
    )


def add_city_identity(data: GraphData, city_index: int) -> Tensor:
    identity = data.x.new_zeros((data.num_nodes, len(DEFAULT_STUDIES)))
    identity[:, city_index] = 1.0
    return torch.cat((data.x, identity), dim=1)


def fit_context_hgb(
    x: np.ndarray, y: np.ndarray, seed: int
) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        learning_rate=0.035,
        max_iter=400,
        max_leaf_nodes=15,
        min_samples_leaf=25,
        l2_regularization=3.0,
        random_state=seed,
    ).fit(x, y)


def hgb_design(data: GraphData, city_index: int) -> np.ndarray:
    labelled = data.building_mask.numpy()
    city = np.zeros((int(labelled.sum()), len(DEFAULT_STUDIES)), dtype=np.float32)
    city[:, city_index] = 1.0
    return np.column_stack((data.x.numpy()[labelled], city))


def append_hgb_residual_problem(
    data: GraphData,
    base_prediction: np.ndarray,
    include_context_covariates: bool,
) -> GraphData:
    labelled_nodes = np.flatnonzero(data.building_mask.numpy())
    if len(base_prediction) != len(labelled_nodes):
        raise ValueError("HGB prediction count differs from labelled nodes")
    node_prediction = torch.zeros(data.num_nodes, dtype=data.x.dtype)
    node_prediction[labelled_nodes] = torch.as_tensor(
        base_prediction, dtype=data.x.dtype
    )
    residual_target = data.y.clone()
    residual_target[labelled_nodes] -= node_prediction[labelled_nodes]
    model_x = (
        torch.cat((data.x, node_prediction[:, None]), dim=1)
        if include_context_covariates
        else node_prediction[:, None]
    )
    feature_names = (
        (*data.feature_names, "nested_context_hgb_prediction")
        if include_context_covariates
        else ("nested_context_hgb_prediction",)
    )
    return replace(
        data,
        x=model_x,
        y=residual_target,
        feature_names=feature_names,
    )


def build_hgb_residual_problem(
    target: GraphData,
    target_split: GroupSplit,
    target_folds: np.ndarray,
    target_city_index: int,
    auxiliary: GraphData | None,
    auxiliary_folds: np.ndarray | None,
    auxiliary_city_index: int | None,
    seed: int,
    include_context_covariates: bool,
) -> tuple[GraphData, GraphData | None, np.ndarray]:
    target_labelled = target.building_mask.numpy()
    target_train = target_split.train.numpy()[target_labelled]
    x_blocks = [hgb_design(target, target_city_index)[target_train]]
    y_blocks = [target.y.numpy()[target_labelled][target_train]]
    fold_blocks = [target_folds[target_train]]

    if auxiliary is not None:
        if auxiliary_folds is None or auxiliary_city_index is None:
            raise ValueError("auxiliary fold and city metadata are required")
        auxiliary_labelled = auxiliary.building_mask.numpy()
        x_blocks.append(hgb_design(auxiliary, auxiliary_city_index))
        y_blocks.append(auxiliary.y.numpy()[auxiliary_labelled])
        fold_blocks.append(auxiliary_folds)

    train_x = np.concatenate(x_blocks)
    train_y = np.concatenate(y_blocks)
    train_fold = np.concatenate(fold_blocks)
    cross_fitted = np.full(len(train_y), np.nan, dtype=float)
    for hold_fold in sorted(int(value) for value in np.unique(train_fold)):
        held = train_fold == hold_fold
        fitted = fit_context_hgb(train_x[~held], train_y[~held], seed + hold_fold)
        cross_fitted[held] = fitted.predict(train_x[held])
    if not np.isfinite(cross_fitted).all():
        raise RuntimeError("incomplete cross-fitted HGB training prediction")

    full_hgb = fit_context_hgb(train_x, train_y, seed + 100)
    target_x = hgb_design(target, target_city_index)
    target_base = full_hgb.predict(target_x)
    target_train_count = int(target_train.sum())
    target_base[target_train] = cross_fitted[:target_train_count]
    target_problem = append_hgb_residual_problem(
        target, target_base, include_context_covariates
    )

    auxiliary_problem = None
    if auxiliary is not None:
        auxiliary_count = int(auxiliary.building_mask.sum())
        auxiliary_base = cross_fitted[-auxiliary_count:]
        auxiliary_problem = append_hgb_residual_problem(
            auxiliary, auxiliary_base, include_context_covariates
        )
    return target_problem, auxiliary_problem, target_base


def combine_with_auxiliary_city(
    target: GraphData,
    target_split: GroupSplit,
    target_city_index: int,
    auxiliary: GraphData | None,
    auxiliary_city_index: int | None,
) -> tuple[GraphData, GroupSplit]:
    target_x = add_city_identity(target, target_city_index)
    if auxiliary is None:
        feature_names = (*target.feature_names, *DEFAULT_STUDIES)
        return (
            GraphData(
                x=target_x,
                node_mass=target.node_mass,
                edge_index=target.edge_index,
                edge_impedance=target.edge_impedance,
                edge_type=target.edge_type,
                y=target.y,
                building_mask=target.building_mask,
                group=target.group,
                label_group=target.label_group,
                sample_weight=target.sample_weight,
                feature_names=feature_names,
            ),
            target_split,
        )
    if auxiliary_city_index is None:
        raise ValueError("auxiliary_city_index is required with auxiliary data")
    if target.feature_names != auxiliary.feature_names:
        raise ValueError("target and auxiliary feature schemas differ")
    if target.num_relations != auxiliary.num_relations:
        raise ValueError("target and auxiliary relation schemas differ")

    auxiliary_x = add_city_identity(auxiliary, auxiliary_city_index)
    node_offset = target.num_nodes
    auxiliary_edges = auxiliary.edge_index + node_offset
    auxiliary_group = auxiliary.group.clone()
    group_offset = int(target.group[target.building_mask].max()) + 1
    auxiliary_group[auxiliary_group >= 0] += group_offset
    combined = GraphData(
        x=torch.cat((target_x, auxiliary_x), dim=0),
        node_mass=torch.cat((target.node_mass, auxiliary.node_mass), dim=0),
        edge_index=torch.cat((target.edge_index, auxiliary_edges), dim=1),
        edge_impedance=torch.cat(
            (target.edge_impedance, auxiliary.edge_impedance), dim=0
        ),
        edge_type=torch.cat((target.edge_type, auxiliary.edge_type), dim=0),
        y=torch.cat((target.y, auxiliary.y), dim=0),
        building_mask=torch.cat(
            (target.building_mask, auxiliary.building_mask), dim=0
        ),
        group=torch.cat((target.group, auxiliary_group), dim=0),
        feature_names=(*target.feature_names, *DEFAULT_STUDIES),
    )
    auxiliary_train = auxiliary.building_mask
    auxiliary_unassigned = torch.full(
        (auxiliary.num_nodes,), -1, dtype=torch.long
    )
    auxiliary_unassigned[auxiliary_train] = 0
    split = GroupSplit(
        train=torch.cat((target_split.train, auxiliary_train), dim=0),
        validation=torch.cat(
            (target_split.validation, torch.zeros_like(auxiliary_train)), dim=0
        ),
        test=torch.cat(
            (target_split.test, torch.zeros_like(auxiliary_train)), dim=0
        ),
        partition=torch.cat(
            (target_split.partition, auxiliary_unassigned), dim=0
        ),
    )
    return combined, split


def mean_fold_metrics(folds: list[dict[str, object]]) -> dict[str, float]:
    return {
        name: float(np.mean([float(fold[name]) for fold in folds]))
        for name in ("r2", "rmse", "mae")
    }


def main() -> None:
    args = build_parser().parse_args()
    studies = tuple(args.study or DEFAULT_STUDIES)
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    excluded_relations = tuple(
        value.strip() for value in args.exclude_relations.split(",") if value.strip()
    )
    relation_access_shifts = parse_relation_access_shifts(
        args.relation_access_shifts
    )
    if not seeds:
        raise ValueError("at least one seed is required")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if excluded_relations and args.export_relation_access:
        raise ValueError(
            "relation-access export is unavailable when relations are excluded"
        )

    loaded = {
        study: load_study(args.case_root, study, args.target)
        for study in DEFAULT_STUDIES
    }
    reference_relations = loaded[DEFAULT_STUDIES[0]][1]
    if loaded[DEFAULT_STUDIES[1]][1] != reference_relations:
        raise ValueError("case studies use different relation schemas")
    unknown_exclusions = sorted(set(excluded_relations) - set(reference_relations))
    if unknown_exclusions:
        raise ValueError(
            f"unknown excluded relations: {', '.join(unknown_exclusions)}"
        )
    if excluded_relations:
        loaded = {
            study: (
                exclude_relation_edges(data, relations, excluded_relations),
                relations,
                lsoa_codes,
                folds,
            )
            for study, (data, relations, lsoa_codes, folds) in loaded.items()
        }

    result: dict[str, object] = {
        "settings": {
            "model": "relation-preserving heterogeneous facility Gravity-GNN",
            "target": args.target,
            "studies": studies,
            "seeds": seeds,
            "hgb_seed": args.hgb_seed,
            "test_folds": args.test_fold or "all",
            "kernel_mode": args.kernel_mode,
            "edge_basis": args.edge_basis,
            "cross_city_pooling": args.cross_city_pooling,
            "queen_message_passing": args.queen_message_passing,
            "hidden_dim": args.hidden_dim,
            "message_dim": args.message_dim,
            "layers": args.layers,
            "area_backbone": args.area_backbone,
            "outer_refit": args.outer_refit,
            "hgb_residual": args.hgb_residual,
            "residual_context_covariates": args.residual_context_covariates,
            "hgb_residual_blend_grid": [0.0, 0.25, 0.5, 0.75, 1.0],
            "dropout": args.dropout,
            "epochs": args.epochs,
            "patience": args.patience,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "device": args.device,
            "validation": (
                "frozen five-fold spatial CV; next fold early stopping; target-city "
                "cross-partition Queen edges removed; auxiliary city is training only"
            ),
            "relations": reference_relations,
            "export_relation_access": args.export_relation_access,
            "relation_access_shifts": relation_access_shifts,
            "relation_access_reference_minutes": 40.0,
            "excluded_relations": excluded_relations,
        },
        "within_city": {},
        "prediction_outputs": {},
    }
    args.prediction_dir.mkdir(parents=True, exist_ok=True)

    for study in studies:
        target, relations, lsoa_codes, folds = loaded[study]
        other_study = next(value for value in DEFAULT_STUDIES if value != study)
        auxiliary = loaded[other_study][0] if args.cross_city_pooling else None
        selected_folds = sorted(
            args.test_fold
            if args.test_fold is not None
            else [int(value) for value in np.unique(folds)]
        )
        target_labelled_nodes = np.flatnonzero(target.building_mask.numpy())
        seed_predictions: list[np.ndarray] = []
        seed_relation_access: list[np.ndarray] = []
        seed_records: list[dict[str, object]] = []
        selection_hgb_cache: dict[
            int, tuple[GraphData, GraphData | None, np.ndarray]
        ] = {}
        outer_hgb_cache: dict[
            int, tuple[GraphData, GraphData | None, np.ndarray]
        ] = {}
        for seed in seeds:
            prediction = np.full(len(lsoa_codes), np.nan, dtype=float)
            relation_access = (
                np.full(
                    (
                        len(relations) - 1,
                        len(lsoa_codes),
                        len(relation_access_shifts),
                    ),
                    np.nan,
                    dtype=np.float32,
                )
                if args.export_relation_access
                else None
            )
            fold_records: list[dict[str, object]] = []
            for test_fold in selected_folds:
                strict_target, target_split, validation_fold = target_fold_split(
                    target, folds, test_fold
                )
                model_target = strict_target
                model_auxiliary = auxiliary
                target_base_prediction = np.zeros(len(lsoa_codes), dtype=float)
                if args.hgb_residual:
                    if test_fold not in selection_hgb_cache:
                        selection_hgb_cache[test_fold] = build_hgb_residual_problem(
                            strict_target,
                            target_split,
                            folds,
                            DEFAULT_STUDIES.index(study),
                            auxiliary,
                            (
                                loaded[other_study][3]
                                if auxiliary is not None
                                else None
                            ),
                            (
                                DEFAULT_STUDIES.index(other_study)
                                if auxiliary is not None
                                else None
                            ),
                            args.hgb_seed + test_fold * 1000,
                            args.residual_context_covariates,
                        )
                    (
                        model_target,
                        model_auxiliary,
                        target_base_prediction,
                    ) = selection_hgb_cache[test_fold]
                combined, combined_split = combine_with_auxiliary_city(
                    model_target,
                    target_split,
                    DEFAULT_STUDIES.index(study),
                    model_auxiliary,
                    (
                        DEFAULT_STUDIES.index(other_study)
                        if model_auxiliary is not None
                        else None
                    ),
                )
                run_seed = seed + test_fold
                seed_everything(run_seed)
                model = HeterogeneousFacilityGravityGNN(
                    input_dim=combined.num_features,
                    hidden_dim=args.hidden_dim,
                    num_relations=len(relations),
                    layers=args.layers,
                    dropout=args.dropout,
                    kernel_mode=args.kernel_mode,
                    edge_basis=args.edge_basis,
                    message_dim=args.message_dim,
                    queen_message_passing=args.queen_message_passing,
                    area_backbone=args.area_backbone,
                )
                fitted = fit_model(
                    model,
                    combined,
                    combined_split,
                    epochs=args.epochs,
                    patience=args.patience,
                    learning_rate=args.learning_rate,
                    weight_decay=args.weight_decay,
                    seed=run_seed,
                    device=args.device,
                )
                residual_weight = 1.0
                if args.hgb_residual:
                    validation_rows = folds == validation_fold
                    inner_residual = fitted.predictions.numpy()[
                        target_labelled_nodes
                    ][validation_rows]
                    validation_y = target.y.numpy()[target_labelled_nodes][
                        validation_rows
                    ]
                    validation_base = target_base_prediction[validation_rows]
                    blend_grid = (0.0, 0.25, 0.5, 0.75, 1.0)
                    residual_weight = min(
                        blend_grid,
                        key=lambda weight: metrics(
                            validation_y,
                            validation_base + weight * inner_residual,
                        )["rmse"],
                    )
                evaluation_predictions = fitted.predictions
                evaluation_base_prediction = target_base_prediction
                evaluation_model = model
                evaluation_graph = combined
                evaluation_train_mask = combined_split.train
                refit_seconds = 0.0
                if args.outer_refit:
                    outer_target, outer_target_split = target_outer_split(
                        target, folds, test_fold
                    )
                    outer_model_target = outer_target
                    outer_model_auxiliary = auxiliary
                    outer_base_prediction = np.zeros(len(lsoa_codes), dtype=float)
                    if args.hgb_residual:
                        if test_fold not in outer_hgb_cache:
                            outer_hgb_cache[test_fold] = build_hgb_residual_problem(
                                outer_target,
                                outer_target_split,
                                folds,
                                DEFAULT_STUDIES.index(study),
                                auxiliary,
                                (
                                    loaded[other_study][3]
                                    if auxiliary is not None
                                    else None
                                ),
                                (
                                    DEFAULT_STUDIES.index(other_study)
                                    if auxiliary is not None
                                    else None
                                ),
                                args.hgb_seed + test_fold * 1000 + 500,
                                args.residual_context_covariates,
                            )
                        (
                            outer_model_target,
                            outer_model_auxiliary,
                            outer_base_prediction,
                        ) = outer_hgb_cache[test_fold]
                    outer_combined, outer_combined_split = combine_with_auxiliary_city(
                        outer_model_target,
                        outer_target_split,
                        DEFAULT_STUDIES.index(study),
                        outer_model_auxiliary,
                        (
                            DEFAULT_STUDIES.index(other_study)
                            if outer_model_auxiliary is not None
                            else None
                        ),
                    )
                    seed_everything(run_seed)
                    model = HeterogeneousFacilityGravityGNN(
                        input_dim=outer_combined.num_features,
                        hidden_dim=args.hidden_dim,
                        num_relations=len(relations),
                        layers=args.layers,
                        dropout=args.dropout,
                        kernel_mode=args.kernel_mode,
                        edge_basis=args.edge_basis,
                        message_dim=args.message_dim,
                        queen_message_passing=args.queen_message_passing,
                        area_backbone=args.area_backbone,
                    )
                    refitted = fit_model_fixed_epochs(
                        model,
                        outer_combined,
                        outer_combined_split.train,
                        epochs=fitted.best_epoch,
                        learning_rate=args.learning_rate,
                        weight_decay=args.weight_decay,
                        seed=run_seed,
                        device=args.device,
                    )
                    evaluation_predictions = refitted.predictions
                    evaluation_base_prediction = outer_base_prediction
                    evaluation_model = model
                    evaluation_graph = outer_combined
                    evaluation_train_mask = outer_combined_split.train
                    refit_seconds = refitted.training_seconds
                test_rows = folds == test_fold
                target_residual_prediction = evaluation_predictions.numpy()[
                    target_labelled_nodes
                ]
                fold_prediction = (
                    evaluation_base_prediction[test_rows]
                    + residual_weight * target_residual_prediction[test_rows]
                )
                prediction[test_rows] = fold_prediction
                if relation_access is not None:
                    fold_access = relation_access_counterfactuals(
                        evaluation_model,
                        evaluation_graph,
                        evaluation_train_mask,
                        target_node_limit=target.num_nodes,
                        target_labelled_nodes=target_labelled_nodes,
                        relation_count=len(relations) - 1,
                        shifts=relation_access_shifts,
                        residual_weight=residual_weight,
                        device=args.device,
                    )
                    relation_access[:, test_rows, :] = fold_access[:, test_rows, :]
                original_target = target.y.numpy()[target_labelled_nodes][test_rows]
                fold_record: dict[str, object] = {
                    "test_fold": test_fold,
                    "validation_fold": validation_fold,
                    "best_epoch": fitted.best_epoch,
                    "selection_training_seconds": fitted.training_seconds,
                    "refit_training_seconds": refit_seconds,
                    "selected_residual_weight": residual_weight,
                    "kernel": model.kernel_summary(),
                    "base_hgb": metrics(
                        original_target,
                        evaluation_base_prediction[test_rows],
                    ),
                    **metrics(original_target, fold_prediction),
                }
                fold_records.append(fold_record)
                print(
                    json.dumps(
                        {
                            "event": "fold_complete",
                            "study": study,
                            "seed": seed,
                            **fold_record,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            seed_predictions.append(prediction)
            if relation_access is not None:
                seed_relation_access.append(relation_access)
            seed_records.append(
                {
                    "seed": seed,
                    "folds": fold_records,
                    "macro_selected_folds": mean_fold_metrics(fold_records),
                }
            )

        stacked = np.stack(seed_predictions)
        available_seed_count = np.sum(np.isfinite(stacked), axis=0)
        ensemble = np.full(stacked.shape[1], np.nan, dtype=float)
        has_prediction = available_seed_count > 0
        ensemble[has_prediction] = (
            np.nansum(stacked[:, has_prediction], axis=0)
            / available_seed_count[has_prediction]
        )
        observed = target.y.numpy()[target_labelled_nodes]
        available = np.isfinite(ensemble)
        ensemble_folds = [
            {
                "fold": fold,
                **metrics(
                    observed[folds == fold],
                    ensemble[folds == fold],
                ),
            }
            for fold in selected_folds
        ]
        city_result: dict[str, object] = {
            "nodes_target": target.num_nodes,
            "nodes_training_graph": (
                target.num_nodes + auxiliary.num_nodes
                if auxiliary is not None
                else target.num_nodes
            ),
            "lsoas": len(lsoa_codes),
            "edges_target": target.num_edges,
            "seed_runs": seed_records,
            "ensemble_folds": ensemble_folds,
            "ensemble_macro_selected_folds": mean_fold_metrics(ensemble_folds),
            "ensemble_selected_rows": metrics(observed[available], ensemble[available]),
        }
        if available.all():
            city_result["ensemble_pooled_oof"] = metrics(observed, ensemble)
        result["within_city"][study] = city_result

        prediction_path = args.prediction_dir / f"{study}.npz"
        np.savez_compressed(
            prediction_path,
            lsoa_codes=lsoa_codes,
            observed=observed,
            fold=folds,
            seed=np.asarray(seeds),
            seed_predictions=stacked,
            ensemble_prediction=ensemble,
            relation_names=np.asarray(relations[:-1]),
            relation_access_shift_minutes=np.asarray(relation_access_shifts),
            seed_relation_access=(
                np.stack(seed_relation_access)
                if seed_relation_access
                else np.empty((0, len(relations) - 1, len(lsoa_codes), 0))
            ),
            ensemble_relation_access=(
                np.nanmean(np.stack(seed_relation_access), axis=0)
                if seed_relation_access
                else np.empty((len(relations) - 1, len(lsoa_codes), 0))
            ),
        )
        result["prediction_outputs"][study] = {
            "path": str(prediction_path.resolve()),
            "sha256": sha256(prediction_path),
            "complete_oof": bool(available.all()),
            "relation_access_exported": bool(seed_relation_access),
            "relation_access_definition": (
                "outer-fold fitted GNN prediction at a 40-minute within-relation "
                "reference minus prediction after the stated time shift, clipped "
                "at 40 minutes; context and all other relations held fixed"
            ),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                study: values["ensemble_macro_selected_folds"]
                for study, values in result["within_city"].items()
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
