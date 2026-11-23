from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import torch

from gravity_gnn.case_studies import sha256
from gravity_gnn.data import (
    GraphData,
    GroupSplit,
    load_graph_npz,
    strict_partition_edge_mask,
)
from gravity_gnn.models import GravityPotentialRegressor
from gravity_gnn.training import fit_model, seed_everything


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the routed facility-to-LSOA learnable Gravity-GNN"
    )
    parser.add_argument(
        "--case-root", type=Path, default=Path("data/processed/case_studies")
    )
    parser.add_argument("--study", action="append", required=True)
    parser.add_argument("--seeds", default="20260826,20260827,20260828")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--graph-residual",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use leakage-controlled Queen-neighbour message passing after gravity aggregation.",
    )
    parser.add_argument("--learning-rate", type=float, default=0.004)
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output", type=Path, default=Path("results/facility_gravity_gnn.json")
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
    result = {}
    for name in ("r2", "rmse", "mae"):
        values = np.asarray([row[name] for row in rows], dtype=float)
        result[name] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)),
        }
    return result


def strict_fold_split(
    data: GraphData, folds: np.ndarray, test_fold: int
) -> tuple[GraphData, GroupSplit, int]:
    unique_folds = sorted(int(value) for value in np.unique(folds))
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


def main() -> None:
    args = build_parser().parse_args()
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    if not seeds:
        raise ValueError("at least one seed is required")
    result = {
        "settings": {
            "studies": args.study,
            "seeds": seeds,
            "model": (
                "explicit relation-specific learnable gravity encoder plus "
                "leakage-controlled Queen-neighbour GNN residual"
                if args.graph_residual
                else "explicit relation-specific learnable gravity encoder plus MLP readout"
            ),
            "facility_source_mass": True,
            "maximum_impedance_minutes": 40.0,
            "target_derived_features": False,
            "absolute_coordinate_features": False,
            "validation": (
                "five frozen outer spatial folds; next fold early stopping; "
                "cross-partition LSOA Queen edges removed; facility-to-LSOA edges retained"
            ),
            "epochs": args.epochs,
            "patience": args.patience,
            "hidden_dim": args.hidden_dim,
            "layers": args.layers,
            "dropout": args.dropout,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "device": args.device,
            "graph_residual": args.graph_residual,
        },
        "within_city": {},
        "oof_outputs": {},
    }
    args.oof_dir.mkdir(parents=True, exist_ok=True)

    for study in args.study:
        case_dir = args.case_root / study
        data = load_graph_npz(case_dir / "facility_lsoa_graph.npz")
        payload = np.load(case_dir / "facility_lsoa_graph.npz", allow_pickle=False)
        relations = payload["relation_names"].astype(str).tolist()
        lsoa_codes = payload["lsoa_codes"].astype(str)
        fold_plan = json.loads(
            (case_dir / "spatial_folds_5.json").read_text(encoding="utf-8")
        )
        folds = np.asarray([fold_plan["lsoa_to_fold"][code] for code in lsoa_codes])
        n_lsoa = int(data.building_mask.sum())
        seed_predictions = []
        seed_records = []
        for seed in seeds:
            prediction = np.full(n_lsoa, np.nan, dtype=float)
            fold_records = []
            for test_fold in sorted(int(value) for value in np.unique(folds)):
                strict_data, split, validation_fold = strict_fold_split(
                    data, folds, test_fold
                )
                seed_everything(seed + test_fold)
                model = GravityPotentialRegressor(
                    input_dim=data.num_features,
                    hidden_dim=args.hidden_dim,
                    num_relations=len(relations),
                    layers=args.layers,
                    dropout=args.dropout,
                    learnable_kernel=True,
                    graph_residual=args.graph_residual,
                    use_source_mass=True,
                )
                model.accessibility.maximum_impedance = 40.0
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
                fold_prediction = fitted.predictions.numpy()[:n_lsoa][test]
                prediction[test] = fold_prediction
                fold_records.append(
                    {
                        "test_fold": test_fold,
                        "validation_fold": validation_fold,
                        "best_epoch": fitted.best_epoch,
                        "training_seconds": fitted.training_seconds,
                        "kernel": model.kernel_summary(),
                        **metrics(data.y.numpy()[:n_lsoa][test], fold_prediction),
                    }
                )
                print(
                    json.dumps(
                        {
                            "event": "fold_complete",
                            "study": study,
                            "seed": seed,
                            **fold_records[-1],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            if not np.isfinite(prediction).all():
                raise RuntimeError(f"incomplete OOF prediction: {study}/{seed}")
            seed_predictions.append(prediction)
            seed_records.append(
                {
                    "seed": seed,
                    "folds": fold_records,
                    "macro": macro_metrics(fold_records),
                    "pooled_oof": metrics(data.y.numpy()[:n_lsoa], prediction),
                }
            )
        ensemble = np.mean(seed_predictions, axis=0)
        ensemble_folds = [
            {
                "fold": fold,
                **metrics(
                    data.y.numpy()[:n_lsoa][folds == fold],
                    ensemble[folds == fold],
                ),
            }
            for fold in sorted(int(value) for value in np.unique(folds))
        ]
        result["within_city"][study] = {
            "nodes": data.num_nodes,
            "lsoas": n_lsoa,
            "edges": data.num_edges,
            "relations": relations,
            "seed_runs": seed_records,
            "ensemble_folds": ensemble_folds,
            "ensemble_macro": macro_metrics(ensemble_folds),
            "ensemble_pooled_oof": metrics(data.y.numpy()[:n_lsoa], ensemble),
        }
        oof_path = args.oof_dir / f"{study}.parquet"
        oof = gpd.read_parquet(oof_path)
        if not np.array_equal(oof["lsoa21cd"].astype(str).to_numpy(), lsoa_codes):
            raise ValueError(f"OOF row order differs for {study}")
        oof["prediction_facility_gravity_gnn"] = ensemble
        oof["residual_facility_gravity_gnn"] = (
            oof["observed"].to_numpy(dtype=float) - ensemble
        )
        oof.to_parquet(oof_path, index=False)
        result["oof_outputs"][study] = {
            "path": str(oof_path.resolve()),
            "rows": len(oof),
            "sha256": sha256(oof_path),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                study: {
                    "r2": values["ensemble_macro"]["r2"]["mean"],
                    "rmse": values["ensemble_macro"]["rmse"]["mean"],
                }
                for study, values in result["within_city"].items()
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
