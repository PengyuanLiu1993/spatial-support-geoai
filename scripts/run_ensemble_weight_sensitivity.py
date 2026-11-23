from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


STUDIES = ("greater_london", "greater_manchester")
STUDY_LABELS = {
    "greater_london": "Greater London",
    "greater_manchester": "Greater Manchester",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Post-hoc sensitivity analysis for the TabPFN weight in the "
            "Heterogeneous Facility Graph Ensemble"
        )
    )
    parser.add_argument(
        "--graph-dir",
        type=Path,
        default=Path("results/heterogeneous_facility_predictions_gated_gravity_cv5"),
    )
    parser.add_argument(
        "--tabpfn-dir",
        type=Path,
        default=Path("results/tabpfn_gravity/screen_v2_e1_oof"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/publication_outputs_heterogeneous_gnn"),
    )
    parser.add_argument("--step", type=float, default=0.01)
    return parser


def metrics(observed: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    residual = prediction - observed
    total = np.sum((observed - observed.mean()) ** 2)
    return {
        "r2": float(1.0 - np.sum(residual**2) / max(float(total), 1e-12)),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluate_weight_grid(
    observed: np.ndarray,
    tabpfn: np.ndarray,
    graph: np.ndarray,
    folds: np.ndarray,
    weights: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_rows: list[dict[str, float | int]] = []
    summary_rows: list[dict[str, float]] = []
    fold_values = sorted(np.unique(folds))
    for weight in weights:
        prediction = weight * tabpfn + (1.0 - weight) * graph
        pooled = metrics(observed, prediction)
        evaluations = []
        for fold in fold_values:
            selected = folds == fold
            row = metrics(observed[selected], prediction[selected])
            evaluations.append(row)
            fold_rows.append(
                {
                    "tabpfn_weight": float(weight),
                    "graph_weight": float(1.0 - weight),
                    "fold": int(fold),
                    **row,
                }
            )
        summary_rows.append(
            {
                "tabpfn_weight": float(weight),
                "graph_weight": float(1.0 - weight),
                **{
                    f"macro_{name}": float(np.mean([row[name] for row in evaluations]))
                    for name in ("r2", "rmse", "mae")
                },
                **{
                    f"fold_sd_{name}": float(
                        np.std([row[name] for row in evaluations], ddof=1)
                    )
                    for name in ("r2", "rmse", "mae")
                },
                **{f"pooled_{name}": pooled[name] for name in ("r2", "rmse", "mae")},
            }
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(fold_rows)


def analytic_least_squares_weight(
    observed: np.ndarray, tabpfn: np.ndarray, graph: np.ndarray
) -> float:
    difference = tabpfn - graph
    denominator = float(difference @ difference)
    if denominator <= 1e-12:
        return 0.5
    return float(np.clip((difference @ (observed - graph)) / denominator, 0.0, 1.0))


def row_at_weight(frame: pd.DataFrame, weight: float) -> pd.Series:
    index = (frame["tabpfn_weight"] - weight).abs().idxmin()
    return frame.loc[index]


def format_interval(values: tuple[float, float]) -> str:
    return f"{values[0]:.2f}--{values[1]:.2f}"


def make_table(
    curves: dict[str, pd.DataFrame], summaries: dict[str, dict[str, object]]
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for study in STUDIES:
        curve = curves[study]
        best_weight = float(summaries[study]["best_macro_rmse_weight"])
        settings = [
            ("Graph component only", 0.0),
            ("25:75", 0.25),
            ("Equal weight (50:50)", 0.50),
            ("75:25", 0.75),
            ("TabPFN component only", 1.0),
            ("Post-hoc city optimum", best_weight),
        ]
        seen: set[tuple[str, float]] = set()
        for setting, weight in settings:
            key = (setting, round(weight, 6))
            if key in seen:
                continue
            seen.add(key)
            row = row_at_weight(curve, weight)
            rows.append(
                {
                    "Case study": STUDY_LABELS[study],
                    "Weight setting": setting,
                    "TabPFN weight": float(row["tabpfn_weight"]),
                    "Graph weight": float(row["graph_weight"]),
                    "R2": float(row["macro_r2"]),
                    "RMSE": float(row["macro_rmse"]),
                    "MAE": float(row["macro_mae"]),
                }
            )
    return pd.DataFrame(rows)


def write_markdown_table(table: pd.DataFrame, path: Path) -> None:
    display = table.copy()
    for column in ("TabPFN weight", "Graph weight"):
        display[column] = display[column].map(lambda value: f"{value:.2f}")
    for column in ("R2", "RMSE", "MAE"):
        display[column] = display[column].map(lambda value: f"{value:.3f}")
    columns = list(display.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in display.iterrows():
        lines.append(
            "| " + " | ".join(str(row[column]) for column in columns) + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex_table(table: pd.DataFrame, path: Path) -> None:
    display = table.rename(columns={"R2": "$R^2$"}).copy()
    latex = display.to_latex(
        index=False,
        float_format=lambda value: f"{value:.3f}",
        escape=False,
        caption="Sensitivity of HFGE performance to the TabPFN ensemble weight.",
        label="tab:hgge_weight_sensitivity",
    )
    path.write_text(latex, encoding="utf-8")


def plot_curves(
    curves: dict[str, pd.DataFrame],
    summaries: dict[str, dict[str, object]],
    output_dir: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
        }
    )
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(11.4, 8.2),
        sharex="col",
        layout="constrained",
    )
    colors = {"r2": "#1f78b4", "rmse": "#e66101"}
    for column, study in enumerate(STUDIES):
        curve = curves[study]
        summary = summaries[study]
        weight = curve["tabpfn_weight"].to_numpy(dtype=float)
        for row, metric in enumerate(("r2", "rmse")):
            axis = axes[row, column]
            values = curve[f"macro_{metric}"].to_numpy(dtype=float)
            axis.plot(weight, values, color=colors[metric], linewidth=2.1)
            axis.axvline(
                0.5,
                color="#333333",
                linestyle="--",
                linewidth=1.2,
                label="Equal weight 50:50" if row == 0 else None,
            )
            optimum = float(summary[f"best_macro_{metric}_weight"])
            optimum_row = row_at_weight(curve, optimum)
            optimum_value = float(optimum_row[f"macro_{metric}"])
            equal_row = row_at_weight(curve, 0.5)
            equal_value = float(equal_row[f"macro_{metric}"])
            axis.scatter(
                [optimum],
                [optimum_value],
                color=colors[metric],
                edgecolor="white",
                linewidth=0.8,
                s=48,
                zorder=4,
                label="Post-hoc optimum" if row == 0 else None,
            )
            axis.scatter(
                [0.5],
                [equal_value],
                color="#333333",
                marker="D",
                s=34,
                zorder=4,
            )
            axis.annotate(
                f"optimum w={optimum:.2f}",
                xy=(optimum, optimum_value),
                xytext=(0, 12 if metric == "r2" else -18),
                textcoords="offset points",
                fontsize=9.5,
                ha="center",
                va="bottom" if metric == "r2" else "top",
                bbox={
                    "boxstyle": "square,pad=0.15",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.92,
                },
            )
            axis.grid(axis="y", color="#d9d9d9", linewidth=0.6)
            axis.set_xlim(0.0, 1.0)
            axis.margins(y=0.16)
            axis.set_ylabel("Macro $R^2$" if metric == "r2" else "Macro RMSE")
            if row == 0:
                axis.set_title(STUDY_LABELS[study])
            else:
                axis.set_xlabel("TabPFN weight, w  (graph weight = 1 - w)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=2,
        frameon=False,
        columnspacing=1.8,
        handletextpad=0.6,
    )
    fig.suptitle("HFGE ensemble-weight sensitivity across spatial folds", fontsize=14)
    for suffix in ("png", "pdf"):
        fig.savefig(
            output_dir / f"figure6_ensemble_weight_sensitivity.{suffix}",
            dpi=600 if suffix == "png" else None,
            bbox_inches="tight",
            pad_inches=0.08,
        )
    plt.close(fig)


def main() -> None:
    args = build_parser().parse_args()
    if args.step <= 0 or args.step > 1:
        raise ValueError("--step must be in (0, 1]")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    weights = np.linspace(0.0, 1.0, int(round(1.0 / args.step)) + 1)
    curves: dict[str, pd.DataFrame] = {}
    fold_curves: dict[str, pd.DataFrame] = {}
    summaries: dict[str, dict[str, object]] = {}
    all_curve_rows = []
    all_fold_rows = []

    for study in STUDIES:
        graph_path = args.graph_dir / f"{study}.npz"
        tabpfn_path = args.tabpfn_dir / f"{study}_context_only.csv"
        graph_data = np.load(graph_path, allow_pickle=False)
        tabpfn_data = pd.read_csv(tabpfn_path)
        codes = graph_data["lsoa_codes"].astype(str)
        if not np.array_equal(
            codes, tabpfn_data["lsoa21cd"].astype(str).to_numpy()
        ):
            raise ValueError(f"row order differs for {study}")
        observed = graph_data["observed"].astype(float)
        graph_prediction = graph_data["ensemble_prediction"].astype(float)
        tabpfn_prediction = tabpfn_data["prediction"].to_numpy(dtype=float)
        folds = graph_data["fold"].astype(int)
        if not np.allclose(
            observed, tabpfn_data["observed"].to_numpy(dtype=float)
        ):
            raise ValueError(f"observed targets differ for {study}")

        curve, fold_curve = evaluate_weight_grid(
            observed, tabpfn_prediction, graph_prediction, folds, weights
        )
        curves[study] = curve
        fold_curves[study] = fold_curve
        all_curve_rows.append(curve.assign(study=study))
        all_fold_rows.append(fold_curve.assign(study=study))

        best_r2 = curve.loc[curve["macro_r2"].idxmax()]
        best_rmse = curve.loc[curve["macro_rmse"].idxmin()]
        equal = row_at_weight(curve, 0.5)
        best_rmse_se = float(best_rmse["fold_sd_rmse"] / np.sqrt(len(np.unique(folds))))
        one_se = curve[
            curve["macro_rmse"] <= float(best_rmse["macro_rmse"]) + best_rmse_se
        ]
        one_percent = curve[
            curve["macro_rmse"] <= float(best_rmse["macro_rmse"]) * 1.01
        ]
        fold_optima = []
        for fold in sorted(np.unique(folds)):
            fold_frame = fold_curve[fold_curve["fold"] == fold]
            row = fold_frame.loc[fold_frame["rmse"].idxmin()]
            fold_optima.append(
                {
                    "fold": int(fold),
                    "tabpfn_weight": float(row["tabpfn_weight"]),
                    "rmse": float(row["rmse"]),
                    "r2": float(row["r2"]),
                }
            )
        tabpfn_residual = observed - tabpfn_prediction
        graph_residual = observed - graph_prediction
        summaries[study] = {
            "n": int(len(observed)),
            "best_macro_r2_weight": float(best_r2["tabpfn_weight"]),
            "best_macro_r2": float(best_r2["macro_r2"]),
            "best_macro_rmse_weight": float(best_rmse["tabpfn_weight"]),
            "best_macro_rmse": float(best_rmse["macro_rmse"]),
            "analytic_pooled_least_squares_weight": analytic_least_squares_weight(
                observed, tabpfn_prediction, graph_prediction
            ),
            "equal_weight": {
                "macro_r2": float(equal["macro_r2"]),
                "macro_rmse": float(equal["macro_rmse"]),
                "macro_mae": float(equal["macro_mae"]),
                "r2_loss_vs_posthoc_best": float(
                    best_r2["macro_r2"] - equal["macro_r2"]
                ),
                "rmse_increase_vs_posthoc_best": float(
                    equal["macro_rmse"] - best_rmse["macro_rmse"]
                ),
                "rmse_percent_increase_vs_posthoc_best": float(
                    100.0
                    * (equal["macro_rmse"] - best_rmse["macro_rmse"])
                    / best_rmse["macro_rmse"]
                ),
            },
            "rmse_one_standard_error_interval": [
                float(one_se["tabpfn_weight"].min()),
                float(one_se["tabpfn_weight"].max()),
            ],
            "rmse_within_one_percent_interval": [
                float(one_percent["tabpfn_weight"].min()),
                float(one_percent["tabpfn_weight"].max()),
            ],
            "residual_correlation": float(
                np.corrcoef(tabpfn_residual, graph_residual)[0, 1]
            ),
            "fold_specific_posthoc_optima": fold_optima,
            "inputs": {
                "graph": str(graph_path.resolve()),
                "graph_sha256": sha256(graph_path),
                "tabpfn": str(tabpfn_path.resolve()),
                "tabpfn_sha256": sha256(tabpfn_path),
            },
        }

    full_curve = pd.concat(all_curve_rows, ignore_index=True)
    full_fold_curve = pd.concat(all_fold_rows, ignore_index=True)
    curve_path = args.output_dir / "ensemble_weight_sensitivity_full.csv"
    fold_path = args.output_dir / "ensemble_weight_sensitivity_by_fold.csv"
    full_curve.to_csv(curve_path, index=False)
    full_fold_curve.to_csv(fold_path, index=False)

    shared_curve = pd.DataFrame(
        {
            "tabpfn_weight": weights,
            "mean_macro_r2": np.mean(
                [curves[study]["macro_r2"].to_numpy(dtype=float) for study in STUDIES],
                axis=0,
            ),
            "mean_relative_rmse": np.mean(
                [
                    curves[study]["macro_rmse"].to_numpy(dtype=float)
                    / curves[study]["macro_rmse"].min()
                    for study in STUDIES
                ],
                axis=0,
            ),
        }
    )
    shared_best_r2 = shared_curve.loc[shared_curve["mean_macro_r2"].idxmax()]
    shared_best_rmse = shared_curve.loc[
        shared_curve["mean_relative_rmse"].idxmin()
    ]
    summary = {
        "analysis": (
            "post-hoc sensitivity analysis using frozen five-fold spatial OOF "
            "predictions; it diagnoses robustness but does not constitute nested "
            "selection of the ensemble weight"
        ),
        "blend_definition": (
            "prediction = w * context_TabPFN + (1 - w) * gated_facility_Gravity_GNN_hybrid"
        ),
        "weight_grid": {
            "minimum": 0.0,
            "maximum": 1.0,
            "step": float(args.step),
        },
        "shared_posthoc_optimum": {
            "by_mean_macro_r2": float(shared_best_r2["tabpfn_weight"]),
            "by_mean_relative_rmse": float(shared_best_rmse["tabpfn_weight"]),
        },
        "studies": summaries,
    }
    summary_path = args.output_dir / "ensemble_weight_sensitivity_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    table = make_table(curves, summaries)
    table_csv_path = args.output_dir / "table2_ensemble_weight_sensitivity.csv"
    table_md_path = args.output_dir / "table2_ensemble_weight_sensitivity.md"
    table_tex_path = args.output_dir / "table2_ensemble_weight_sensitivity.tex"
    table.to_csv(table_csv_path, index=False)
    write_markdown_table(table, table_md_path)
    write_latex_table(table, table_tex_path)
    plot_curves(curves, summaries, args.output_dir)

    london = summaries["greater_london"]
    manchester = summaries["greater_manchester"]
    notes = f"""# HGGE 集成权重敏感性分析

## 方法

令 $w$ 为 Context-TabPFN 的预测权重，异构设施 Gravity 图混合模型的权重为 $1-w$：

$$
\\hat y_{{HGGE}}=w\\hat y_{{TabPFN}}+(1-w)\\hat y_{{Gravity-GNN}}.
$$

在冻结的五折连续空间交叉验证 OOF 预测上，将 $w$ 从 0 到 1 按 {args.step:.2f} 递增，重新计算每一折的 $R^2$、RMSE 和 MAE，并对五折作等权宏平均。该分析是事后稳健性诊断，不是严格嵌套的权重选择，因此最优点不得作为无偏泛化性能使用。

## 结果

- 伦敦：宏平均 RMSE 最优权重为 $w={london['best_macro_rmse_weight']:.2f}$，50:50 的 RMSE 仅高 {london['equal_weight']['rmse_increase_vs_posthoc_best']:.3f}（{london['equal_weight']['rmse_percent_increase_vs_posthoc_best']:.2f}%），$R^2$ 损失为 {london['equal_weight']['r2_loss_vs_posthoc_best']:.4f}。RMSE 的 1% 近最优区间为 {format_interval(tuple(london['rmse_within_one_percent_interval']))}。
- 曼城：宏平均 RMSE 最优权重为 $w={manchester['best_macro_rmse_weight']:.2f}$，50:50 的 RMSE 仅高 {manchester['equal_weight']['rmse_increase_vs_posthoc_best']:.3f}（{manchester['equal_weight']['rmse_percent_increase_vs_posthoc_best']:.2f}%），$R^2$ 损失为 {manchester['equal_weight']['r2_loss_vs_posthoc_best']:.4f}。RMSE 的 1% 近最优区间为 {format_interval(tuple(manchester['rmse_within_one_percent_interval']))}。
- 两城平均宏 $R^2$ 的共同事后最优权重为 $w={summary['shared_posthoc_optimum']['by_mean_macro_r2']:.2f}$；按相对 RMSE 选择时为 $w={summary['shared_posthoc_optimum']['by_mean_relative_rmse']:.2f}$。
- 两个分量模型的 OOF 残差相关系数分别为 {london['residual_correlation']:.3f} 和 {manchester['residual_correlation']:.3f}。相关性较高但不等于 1，说明集成增益来自有限但稳定的互补误差。

## 可直接放入论文的方法表述

为检验集成系数的敏感性，本文将 Context-TabPFN 的权重 $w$ 在 $[0,1]$ 区间内以 0.01 为步长变化，并令异构设施 Gravity 图模型的权重为 $1-w$。所有组合均使用相同的五折连续空间交叉验证 OOF 预测计算，避免因重新训练造成的随机差异。主要结果采用 50:50 等权集成；该设定不增加需要估计的元模型参数，并为两个信息来源提供对称贡献。权重扫描仅用于评估结果对该设定的稳健性，而不用于事后替换主结果。

## 可直接放入论文的结果表述

敏感性分析表明，HGGE 的性能在等权设定附近呈平缓变化。伦敦和曼城的事后最优 TabPFN 权重分别为 {london['best_macro_rmse_weight']:.2f} 和 {manchester['best_macro_rmse_weight']:.2f}，但相较各城市的事后最优点，50:50 的宏平均 RMSE 仅分别增加 {london['equal_weight']['rmse_percent_increase_vs_posthoc_best']:.2f}% 和 {manchester['equal_weight']['rmse_percent_increase_vs_posthoc_best']:.2f}%。因此，主结果对权重的小幅改变不敏感，采用透明且无需额外拟合的等权组合是合理的。由于该扫描基于已生成的 OOF 预测，事后最优权重仅作为敏感性参照，而不作为新的泛化性能估计。

## 审稿边界

不能把扫描得到的城市最优权重称为“交叉验证选择的最优权重”。如果论文要将数据学习权重作为正式模型组成部分，需要在每个外层空间折的训练部分重新生成内层 OOF 预测并学习 $w$，然后将冻结权重用于相应外层测试折。
"""
    notes_path = args.output_dir / "ensemble_weight_sensitivity_notes_zh.md"
    notes_path.write_text(notes, encoding="utf-8")

    manifest_paths = [
        curve_path,
        fold_path,
        summary_path,
        table_csv_path,
        table_md_path,
        table_tex_path,
        args.output_dir / "figure6_ensemble_weight_sensitivity.png",
        args.output_dir / "figure6_ensemble_weight_sensitivity.pdf",
        notes_path,
    ]
    manifest = {
        path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in manifest_paths
    }
    manifest_path = args.output_dir / "ensemble_weight_sensitivity_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
