from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .baselines import fit_tabular_baseline
from .data import (
    assert_group_disjoint,
    load_graph_npz,
    make_fold_split,
    make_group_split,
    make_synthetic_graph,
    strict_partition_edge_mask,
    validate_graph,
)
from .models import GELGNN, build_model
from .training import fit_model, seed_everything


DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 20260826,
    "num_buildings": 480,
    "num_amenities": 80,
    "neighbors_per_building": 5,
    "amenities_per_building": 10,
    "hidden_dim": 32,
    "layers": 2,
    "dropout": 0.1,
    "epochs": 120,
    "patience": 20,
    "learning_rate": 0.01,
    "weight_decay": 0.0001,
}


def _load_config(path: str | None) -> dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    if path:
        config.update(json.loads(Path(path).read_text(encoding="utf-8")))
    return config


def _write_json(path: str | None, payload: dict[str, Any]) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True)
    if path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text + "\n", encoding="utf-8")
    print(text)


def run_smoke(args: argparse.Namespace) -> None:
    config = _load_config(args.config)
    if args.epochs is not None:
        config["epochs"] = args.epochs
    data = make_synthetic_graph(
        num_buildings=config["num_buildings"],
        num_amenities=config["num_amenities"],
        neighbors_per_building=config["neighbors_per_building"],
        amenities_per_building=config["amenities_per_building"],
        seed=config["seed"],
    )
    split = make_group_split(data, seed=config["seed"])
    assert_group_disjoint(data, split)
    strict_edge_mask = strict_partition_edge_mask(data, split.partition)
    strict_data = data.with_edge_mask(strict_edge_mask)
    validate_graph(strict_data)

    model_names = ["mlp", "gcn", "gat", "fixed_gravity", "gel_gnn"]
    if args.include_draft_faithful:
        model_names.append("draft_faithful")
    results: dict[str, Any] = {}
    for index, name in enumerate(model_names):
        seed_everything(config["seed"] + index)
        model = build_model(
            name=name,
            input_dim=strict_data.num_features,
            hidden_dim=config["hidden_dim"],
            num_relations=strict_data.num_relations,
            layers=config["layers"],
            dropout=config["dropout"],
        )
        fit = fit_model(
            model,
            strict_data,
            split,
            epochs=config["epochs"],
            patience=min(config["patience"], config["epochs"]),
            learning_rate=config["learning_rate"],
            weight_decay=config["weight_decay"],
            seed=config["seed"] + index,
            device=args.device,
        )
        result: dict[str, Any] = {
            "best_epoch": fit.best_epoch,
            "training_seconds": fit.training_seconds,
            "metrics": fit.metrics,
        }
        if hasattr(model, "kernel_summary"):
            result["kernel"] = model.kernel_summary()
        results[name] = result

    payload = {
        "synthetic_only": True,
        "warning": "This run validates software mechanics; it is not evidence about London or health outcomes.",
        "config": config,
        "graph": {
            "nodes": strict_data.num_nodes,
            "buildings": int(strict_data.building_mask.sum()),
            "edges_before_partition_isolation": data.num_edges,
            "edges_after_partition_isolation": strict_data.num_edges,
            "relations": strict_data.num_relations,
            "spatial_groups": int(torch.unique(strict_data.group[strict_data.building_mask]).numel()),
        },
        "split": {
            "train_buildings": int(split.train.sum()),
            "validation_buildings": int(split.validation.sum()),
            "test_buildings": int(split.test.sum()),
        },
        "models": results,
    }
    _write_json(args.output, payload)


def run_real(args: argparse.Namespace) -> None:
    data = load_graph_npz(args.path)
    if args.target_bundle:
        source = np.load(args.target_bundle, allow_pickle=False)
        if args.target_name not in source.files:
            raise ValueError(
                f"target {args.target_name!r} not found; available: {sorted(source.files)}"
            )
        target = torch.as_tensor(source[args.target_name], dtype=torch.float32)
        if target.shape != data.y.shape:
            raise ValueError("target array must contain one value per graph node")
        if not torch.isfinite(target[data.building_mask]).all():
            raise ValueError("every labelled building must have a finite target")
        data = replace(data, y=target)
    model_names = [name.strip() for name in args.models.split(",") if name.strip()]
    tabular_names = {
        "dummy_mean",
        "ridge",
        "ridge_cv",
        "hist_gradient_boosting",
        "edge_ridge",
        "edge_ridge_cv",
        "gravity_ridge",
        "gravity_ridge_cv",
        "gravity_hist_gradient_boosting",
        "edge_ridge_spatial",
        "gravity_ridge_spatial",
        "hybrid_ridge_spatial",
        "spatial_ridge",
        "context_ridge",
        "mechanism_ridge",
        "reconstruction_ridge",
        "context_gravity_residual",
        "pollution_ridge",
        "pollution_ridge_cv",
        "context_pollution_blend",
        "context_pls",
        "context_bayesian_ridge",
        "context_ridge_spatial",
        "edge_hist_gradient_boosting",
    }
    split_seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    fold_plan: dict[str, Any] | None = None
    planned_splits: list[tuple[str, int, Any]] = []
    if args.fold_plan:
        fold_plan = json.loads(Path(args.fold_plan).read_text(encoding="utf-8"))
        group_to_fold = torch.as_tensor(fold_plan["group_to_fold"], dtype=torch.long)
        requested_folds = (
            {int(value.strip()) for value in args.test_folds.split(",") if value.strip()}
            if args.test_folds
            else None
        )
        for item in fold_plan["fold_runs"]:
            test_fold = int(item["test_fold"])
            if requested_folds is not None and test_fold not in requested_folds:
                continue
            validation_fold = int(item["validation_fold"])
            split = make_fold_split(data, group_to_fold, test_fold, validation_fold)
            planned_splits.append((f"fold_{test_fold}", args.model_seed + 100 * test_fold, split))
    else:
        for split_seed in split_seeds:
            planned_splits.append(
                (f"seed_{split_seed}", split_seed, make_group_split(data, seed=split_seed))
            )
    runs: list[dict[str, Any]] = []
    for split_id, run_seed, split in planned_splits:
        assert_group_disjoint(data, split)
        edge_mask = strict_partition_edge_mask(data, split.partition)
        strict_data = data.with_edge_mask(edge_mask)
        model_results: dict[str, Any] = {}
        for index, name in enumerate(model_names):
            model_seed = run_seed + index
            if name in tabular_names:
                fit = fit_tabular_baseline(name, strict_data, split, seed=model_seed)
                model_results[name] = {
                    "training_seconds": fit.training_seconds,
                    "metrics": fit.metrics,
                }
                continue
            seed_everything(model_seed)
            model = build_model(
                name=name,
                input_dim=strict_data.num_features,
                hidden_dim=args.hidden_dim,
                num_relations=strict_data.num_relations,
                layers=args.layers,
                dropout=args.dropout,
            )
            fit = fit_model(
                model,
                strict_data,
                split,
                epochs=args.epochs,
                patience=min(args.patience, args.epochs),
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                seed=model_seed,
                device=args.device,
            )
            result: dict[str, Any] = {
                "best_epoch": fit.best_epoch,
                "training_seconds": fit.training_seconds,
                "metrics": fit.metrics,
            }
            if hasattr(model, "kernel_summary"):
                result["kernel"] = model.kernel_summary()
            model_results[name] = result
        runs.append(
            {
                "split_id": split_id,
                "model_seed_base": run_seed,
                "edges_before_partition_isolation": data.num_edges,
                "edges_after_partition_isolation": strict_data.num_edges,
                "split": {
                    "train_buildings": int(split.train.sum()),
                    "validation_buildings": int(split.validation.sum()),
                    "test_buildings": int(split.test.sum()),
                    "train_spatial_blocks": int(torch.unique(data.group[split.train]).numel()),
                    "validation_spatial_blocks": int(
                        torch.unique(data.group[split.validation]).numel()
                    ),
                    "test_spatial_blocks": int(torch.unique(data.group[split.test]).numel()),
                    "train_label_groups": int(
                        torch.unique(data.label_group[split.train]).numel()
                    )
                    if data.label_group is not None
                    else None,
                    "validation_label_groups": int(
                        torch.unique(data.label_group[split.validation]).numel()
                    )
                    if data.label_group is not None
                    else None,
                    "test_label_groups": int(
                        torch.unique(data.label_group[split.test]).numel()
                    )
                    if data.label_group is not None
                    else None,
                },
                "models": model_results,
            }
        )
    payload = {
        "real_data": True,
        "target_name": args.target_name,
        "target_interpretation": "LSOA-level target repeated for supervised building nodes; building values are modelled proxies and grouped LSOA metrics are primary.",
        "graph": {
            "path": str(Path(args.path).resolve()),
            "nodes": data.num_nodes,
            "buildings": int(data.building_mask.sum()),
            "edges": data.num_edges,
            "features": data.num_features,
            "relations": data.num_relations,
            "spatial_blocks": int(torch.unique(data.group[data.building_mask]).numel()),
            "label_groups": int(
                torch.unique(data.label_group[data.building_mask]).numel()
            )
            if data.label_group is not None
            else None,
        },
        "settings": {
            "models": model_names,
            "split_strategy": "fixed_spatial_cluster_cv" if fold_plan else "random_spatial_blocks",
            "seeds": None if fold_plan else split_seeds,
            "fold_plan": str(Path(args.fold_plan).resolve()) if args.fold_plan else None,
            "fold_plan_metadata": fold_plan.get("metadata") if fold_plan else None,
            "target_bundle": str(Path(args.target_bundle).resolve()) if args.target_bundle else None,
            "epochs": args.epochs,
            "patience": args.patience,
            "hidden_dim": args.hidden_dim,
            "layers": args.layers,
            "dropout": args.dropout,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "device": args.device,
        },
        "runs": runs,
    }
    _write_json(args.output, payload)


def run_validate(args: argparse.Namespace) -> None:
    data = load_graph_npz(args.path)
    payload = {
        "valid": True,
        "path": str(Path(args.path).resolve()),
        "nodes": data.num_nodes,
        "buildings": int(data.building_mask.sum()),
        "edges": data.num_edges,
        "features": data.num_features,
        "relations": data.num_relations,
        "groups": int(torch.unique(data.group[data.building_mask]).numel()),
    }
    _write_json(None, payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproducible gravity-GNN experiments"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser(
        "smoke", help="run a synthetic mechanism and regression check"
    )
    smoke.add_argument("--config", default="configs/synthetic_smoke.json")
    smoke.add_argument("--epochs", type=int)
    smoke.add_argument("--device", default="cpu")
    smoke.add_argument("--output", default="results/synthetic_smoke.json")
    smoke.add_argument("--include-draft-faithful", action="store_true")
    smoke.set_defaults(function=run_smoke)

    validate = subparsers.add_parser("validate-data", help="validate a real graph bundle")
    validate.add_argument("path")
    validate.set_defaults(function=run_validate)

    real = subparsers.add_parser("real", help="run strict spatial experiments on a graph bundle")
    real.add_argument("path")
    real.add_argument(
        "--models",
        default="ridge,hist_gradient_boosting,mlp,gcn,gat,fixed_gravity,gel_gnn",
    )
    real.add_argument("--seeds", default="20260826")
    real.add_argument("--target-bundle")
    real.add_argument("--target-name", default="ahah")
    real.add_argument(
        "--fold-plan",
        help="JSON plan assigning every spatial group to a fixed fold; overrides --seeds",
    )
    real.add_argument("--model-seed", type=int, default=20260826)
    real.add_argument(
        "--test-folds",
        help="comma-separated test-fold ids from --fold-plan; defaults to every fold",
    )
    real.add_argument("--epochs", type=int, default=80)
    real.add_argument("--patience", type=int, default=20)
    real.add_argument("--hidden-dim", type=int, default=32)
    real.add_argument("--layers", type=int, default=2)
    real.add_argument("--dropout", type=float, default=0.1)
    real.add_argument("--learning-rate", type=float, default=0.01)
    real.add_argument("--weight-decay", type=float, default=0.0001)
    real.add_argument("--device", default="cpu")
    real.add_argument("--output", default="results/camden_pilot.json")
    real.set_defaults(function=run_real)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
