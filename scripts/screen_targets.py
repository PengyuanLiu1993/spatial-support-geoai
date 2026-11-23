from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from gravity_gnn.baselines import fit_tabular_baseline
from gravity_gnn.data import (
    load_graph_npz,
    make_fold_split,
    strict_partition_edge_mask,
)


MODELS = [
    "dummy_mean",
    "ridge",
    "hist_gradient_boosting",
    "edge_ridge",
    "edge_hist_gradient_boosting",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Screen all LSOA targets with cheap fixed spatial-CV baselines"
    )
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--fold-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260826)
    return parser


def mean_sd(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "sd": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
    }


def main() -> None:
    args = build_parser().parse_args()
    data = load_graph_npz(args.graph)
    target_source = np.load(args.targets, allow_pickle=False)
    plan = json.loads(args.fold_plan.read_text(encoding="utf-8"))
    group_to_fold = torch.as_tensor(plan["group_to_fold"], dtype=torch.long)
    splits = []
    for item in plan["fold_runs"]:
        test_fold = int(item["test_fold"])
        validation_fold = int(item["validation_fold"])
        split = make_fold_split(data, group_to_fold, test_fold, validation_fold)
        strict_data = data.with_edge_mask(
            strict_partition_edge_mask(data, split.partition)
        )
        splits.append((test_fold, split, strict_data))

    results: dict[str, object] = {}
    for target_index, target_name in enumerate(target_source.files):
        target = torch.as_tensor(target_source[target_name], dtype=torch.float32)
        target_runs: dict[str, list[dict[str, float | int]]] = {
            model: [] for model in MODELS
        }
        for test_fold, split, strict_data in splits:
            target_data = replace(strict_data, y=target)
            for model_index, model in enumerate(MODELS):
                fit = fit_tabular_baseline(
                    model,
                    target_data,
                    split,
                    seed=args.seed + 1000 * target_index + 100 * test_fold + model_index,
                )
                metrics = fit.metrics["test_grouped"]
                target_runs[model].append(
                    {
                        "fold": test_fold,
                        "r2": metrics["r2"],
                        "rmse": metrics["rmse"],
                        "mae": metrics["mae"],
                    }
                )
        results[target_name] = {
            "runs": target_runs,
            "summary": {
                model: {
                    metric: mean_sd([float(run[metric]) for run in runs])
                    for metric in ("r2", "rmse", "mae")
                }
                for model, runs in target_runs.items()
            },
        }
        best = max(
            MODELS,
            key=lambda model: results[target_name]["summary"][model]["r2"]["mean"],
        )
        print(
            f"{target_name:18s} {best:31s} "
            f"R2={results[target_name]['summary'][best]['r2']['mean']:+.3f}"
        )

    payload = {
        "settings": {
            "graph": str(args.graph.resolve()),
            "targets": str(args.targets.resolve()),
            "fold_plan": str(args.fold_plan.resolve()),
            "models": MODELS,
            "seed": args.seed,
            "primary_metric": "LSOA-grouped test metrics",
        },
        "targets": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
