from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import torch

from gravity_gnn.case_studies import sha256
from gravity_gnn.data import GraphData, GroupSplit, validate_graph
from gravity_gnn.models import build_model
from gravity_gnn.training import fit_model, seed_everything


DEFAULT_MODELS = ("gcn", "gat")
SUPPORTED_MODELS = ("gcn", "gat", "fixed_gravity", "gel_gnn")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run GCN, GAT, and gravity-GNN models on the frozen LSOA graph"
    )
    parser.add_argument(
        "--case-root", type=Path, default=Path("data/processed/case_studies")
    )
    parser.add_argument("--study", action="append", required=True)
    parser.add_argument("--seeds", default="20260826,20260827,20260828")
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--walking-metres-per-minute", type=float, default=80.0)
    parser.add_argument(
        "--output", type=Path, default=Path("results/gnn_baselines.json")
    )
    parser.add_argument(
        "--oof-dir", type=Path, default=Path("results/oof_predictions")
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


def macro_metrics(rows: list[dict]) -> dict:
    output = {}
    for name in ("r2", "rmse", "mae"):
        values = np.asarray([row[name] for row in rows], dtype=float)
        output[name] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)),
        }
    return output


def queen_edges(frame: gpd.GeoDataFrame) -> np.ndarray:
    source = []
    destination = []
    index = frame.sindex
    for node, geometry in enumerate(frame.geometry):
        neighbors = index.query(geometry, predicate="touches")
        for neighbor in neighbors:
            neighbor = int(neighbor)
            if neighbor != node:
                source.append(neighbor)
                destination.append(node)
    if not source:
        raise ValueError("queen-contiguity graph has no edges")
    return np.asarray([source, destination], dtype=np.int64)


def load_case(
    case_root: Path, study: str, walking_metres_per_minute: float
) -> tuple[GraphData, gpd.GeoDataFrame, np.ndarray, list[str]]:
    case_dir = case_root / study
    sources = [
        ("official_context.parquet", "official_context_summary.json"),
        ("osm_context.parquet", "osm_context_summary.json"),
        ("os_green_context.parquet", "os_green_context_summary.json"),
        ("sentinel_ndvi_context.parquet", "sentinel_ndvi_context_summary.json"),
    ]
    frame = gpd.read_parquet(case_dir / sources[0][0])
    feature_names = []
    for position, (parquet_name, summary_name) in enumerate(sources):
        if position:
            extra = gpd.read_parquet(case_dir / parquet_name).drop(columns="geometry")
            frame = frame.merge(extra, on="lsoa21cd", validate="one_to_one")
        summary = json.loads((case_dir / summary_name).read_text(encoding="utf-8"))
        feature_names.extend(summary["features"])
    x = frame[feature_names].to_numpy(dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError(f"{study} contains non-finite node features")
    edge_index = queen_edges(frame)
    points = frame.geometry.representative_point()
    coordinates = np.column_stack((points.x, points.y)).astype(np.float32)
    edge_distance_m = np.linalg.norm(
        coordinates[edge_index[0]] - coordinates[edge_index[1]], axis=1
    )
    edge_minutes = edge_distance_m / walking_metres_per_minute
    building_index = feature_names.index("log_osm_building_count")
    building_count = np.expm1(np.maximum(x[:, building_index].astype(float), 0.0))
    positive = building_count[building_count > 0]
    mass_scale = float(np.median(positive)) if len(positive) else 1.0
    node_mass = np.clip(building_count / max(mass_scale, 1.0), 0.1, 10.0)
    fold_plan = json.loads((case_dir / "spatial_folds_5.json").read_text(encoding="utf-8"))
    folds = frame["lsoa21cd"].map(fold_plan["lsoa_to_fold"]).to_numpy(dtype=np.int64)
    node_count = len(frame)
    data = GraphData(
        x=torch.as_tensor(x),
        node_mass=torch.as_tensor(node_mass, dtype=torch.float32),
        edge_index=torch.as_tensor(edge_index, dtype=torch.long),
        edge_impedance=torch.as_tensor(edge_minutes, dtype=torch.float32),
        edge_type=torch.zeros(edge_index.shape[1], dtype=torch.long),
        y=torch.as_tensor(frame["ahah"].to_numpy(dtype=np.float32)),
        building_mask=torch.ones(node_count, dtype=torch.bool),
        group=torch.arange(node_count, dtype=torch.long),
        feature_names=tuple(feature_names),
    )
    validate_graph(data)
    return data, frame, folds, feature_names


def strict_fold_split(data: GraphData, folds: np.ndarray, test_fold: int) -> tuple[GraphData, GroupSplit, int]:
    unique_folds = sorted(int(value) for value in np.unique(folds))
    position = unique_folds.index(test_fold)
    validation_fold = unique_folds[(position + 1) % len(unique_folds)]
    train = torch.as_tensor((folds != test_fold) & (folds != validation_fold))
    validation = torch.as_tensor(folds == validation_fold)
    test = torch.as_tensor(folds == test_fold)
    partition = torch.full((data.num_nodes,), -1, dtype=torch.long)
    partition[train] = 0
    partition[validation] = 1
    partition[test] = 2
    source, destination = data.edge_index
    edge_mask = partition[source] == partition[destination]
    strict_data = data.with_edge_mask(edge_mask)
    split = GroupSplit(
        train=train,
        validation=validation,
        test=test,
        partition=partition,
    )
    return strict_data, split, validation_fold


def main() -> None:
    args = build_parser().parse_args()
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    models = tuple(name.strip() for name in args.models.split(",") if name.strip())
    if not seeds:
        raise ValueError("at least one seed is required")
    if not models or any(name not in SUPPORTED_MODELS for name in models):
        raise ValueError(f"models must be selected from {SUPPORTED_MODELS}")
    if args.walking_metres_per_minute <= 0:
        raise ValueError("walking-metres-per-minute must be positive")
    result = {
        "settings": {
            "studies": args.study,
            "models": list(models),
            "seeds": list(seeds),
            "graph": "directed symmetric Queen-contiguity LSOA adjacency",
            "edge_impedance": (
                "representative-point Euclidean distance converted to walking minutes; "
                "consumed only by gravity models"
            ),
            "node_mass": (
                "median-scaled OSM building count clipped to [0.1, 10]; "
                "consumed only by gravity models"
            ),
            "walking_metres_per_minute": args.walking_metres_per_minute,
            "absolute_coordinate_features": False,
            "target_derived_features": False,
            "strict_partition_edges": True,
            "validation": (
                "five frozen outer spatial folds; the next fold is a fixed early-stopping "
                "validation fold and edges across train/validation/test partitions are removed"
            ),
            "epochs": args.epochs,
            "patience": args.patience,
            "hidden_dim": args.hidden_dim,
            "layers": args.layers,
            "dropout": args.dropout,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "device": args.device,
        },
        "within_city": {},
        "oof_outputs": {},
    }
    args.oof_dir.mkdir(parents=True, exist_ok=True)

    for study in args.study:
        data, frame, folds, feature_names = load_case(
            args.case_root, study, args.walking_metres_per_minute
        )
        study_result = {
            "nodes": data.num_nodes,
            "directed_edges": data.num_edges,
            "features": len(feature_names),
            "models": {},
        }
        ensemble_predictions = {}
        for model_name in models:
            seed_predictions = []
            seed_records = []
            for seed in seeds:
                prediction = np.full(data.num_nodes, np.nan, dtype=float)
                fold_records = []
                for test_fold in sorted(int(value) for value in np.unique(folds)):
                    strict_data, split, validation_fold = strict_fold_split(
                        data, folds, test_fold
                    )
                    seed_everything(seed + test_fold)
                    model = build_model(
                        model_name,
                        input_dim=data.num_features,
                        hidden_dim=args.hidden_dim,
                        num_relations=1,
                        layers=args.layers,
                        dropout=args.dropout,
                    )
                    fitted = fit_model(
                        model,
                        strict_data,
                        split,
                        epochs=args.epochs,
                        patience=args.patience,
                        learning_rate=args.learning_rate,
                        weight_decay=args.weight_decay,
                        seed=seed + test_fold,
                        device=args.device,
                    )
                    test = folds == test_fold
                    prediction[test] = fitted.predictions.numpy()[test]
                    fold_records.append(
                        {
                            "test_fold": test_fold,
                            "validation_fold": validation_fold,
                            "best_epoch": fitted.best_epoch,
                            "training_seconds": fitted.training_seconds,
                            **metrics(data.y.numpy()[test], prediction[test]),
                        }
                    )
                    kernel_summary = getattr(model, "kernel_summary", None)
                    if kernel_summary is not None:
                        summary = kernel_summary()
                        if summary is not None:
                            fold_records[-1]["kernel"] = summary
                if not np.isfinite(prediction).all():
                    raise RuntimeError(f"incomplete OOF prediction: {study}/{model_name}/{seed}")
                seed_predictions.append(prediction)
                seed_records.append(
                    {
                        "seed": seed,
                        "folds": fold_records,
                        "macro": macro_metrics(fold_records),
                        "pooled_oof": metrics(data.y.numpy(), prediction),
                    }
                )
            ensemble = np.mean(seed_predictions, axis=0)
            ensemble_predictions[model_name] = ensemble
            ensemble_folds = [
                {
                    "fold": fold,
                    **metrics(data.y.numpy()[folds == fold], ensemble[folds == fold]),
                }
                for fold in sorted(int(value) for value in np.unique(folds))
            ]
            study_result["models"][model_name] = {
                "seed_runs": seed_records,
                "ensemble_folds": ensemble_folds,
                "ensemble_macro": macro_metrics(ensemble_folds),
                "ensemble_pooled_oof": metrics(data.y.numpy(), ensemble),
            }
        result["within_city"][study] = study_result

        oof_path = args.oof_dir / f"{study}.parquet"
        if oof_path.exists():
            oof = gpd.read_parquet(oof_path)
            if not np.array_equal(oof["lsoa21cd"].to_numpy(), frame["lsoa21cd"].to_numpy()):
                raise ValueError(f"existing OOF row order differs for {study}")
        else:
            oof = frame[["lsoa21cd", "geometry"]].copy()
            oof["fold"] = folds
            oof["observed"] = data.y.numpy()
        for model_name, prediction in ensemble_predictions.items():
            oof[f"prediction_{model_name}"] = prediction
            oof[f"residual_{model_name}"] = data.y.numpy() - prediction
        oof.to_parquet(oof_path, index=False)
        result["oof_outputs"][study] = {
            "path": str(oof_path.resolve()),
            "sha256": sha256(oof_path),
            "rows": len(oof),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    concise = {
        study: {
            name: {
                "r2": values["ensemble_macro"]["r2"]["mean"],
                "rmse": values["ensemble_macro"]["rmse"]["mean"],
            }
            for name, values in study_result["models"].items()
        }
        for study, study_result in result["within_city"].items()
    }
    print(json.dumps(concise, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
