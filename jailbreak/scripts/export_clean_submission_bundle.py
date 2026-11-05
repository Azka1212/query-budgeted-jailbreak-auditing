"""Create an auditable JSON export and paper-ready figures for the Q=25 rerun."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
TABLE_ROOT = ARTIFACTS / "revision_tables" / "paper_tables"
OUT = ARTIFACTS / "submission_bundle_q25"
FIG = OUT / "figures"
TABLES = OUT / "tables"

METHODS = [
    "GCG",
    "AutoDAN",
    "GPTFuzzer",
    "PAIR",
    "TAP",
    "RLBreaker",
    "xJailbreak",
    "RL-based Multi-agent Framework (Ours)",
]
SHORT = {
    "RL-based Multi-agent Framework (Ours)": "Ours",
}
COLORS = {
    "GCG": "#8c8c8c",
    "AutoDAN": "#6c8ebf",
    "GPTFuzzer": "#82b366",
    "PAIR": "#d79b00",
    "TAP": "#9673a6",
    "RLBreaker": "#b85450",
    "xJailbreak": "#00a2ae",
    "RL-based Multi-agent Framework (Ours)": "#d62728",
}
CORE_TARGETS = [
    ("qwen2_5_14b", "Qwen2.5-14B"),
    ("deepseek_r1_7b", "DeepSeek-R1-7B"),
    ("vicuna_13b", "Vicuna-13B"),
    ("gpt_oss_20b", "GPT-OSS-20B"),
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def raw_metrics() -> list[dict[str, Any]]:
    paths = sorted(
        {
            *ARTIFACTS.glob("**/metrics.json"),
            *ARTIFACTS.glob("**/eval_metrics.json"),
        }
    )
    rows: list[dict[str, Any]] = []
    for path in paths:
        if OUT in path.parents:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            error = None
        except Exception as exc:
            data = None
            error = f"{type(exc).__name__}: {exc}"
        rows.append(
            {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
                "data": data,
                "read_error": error,
            }
        )
    return rows


def table_exports() -> dict[str, list[dict[str, str]]]:
    sources = {
        "main_5d": TABLE_ROOT / "main_5d_results_wide.csv",
        "guard_models": TABLE_ROOT / "GUARD-models_wide.csv",
        "cross_dataset_transfer": TABLE_ROOT / "cross_dataset_transfer.csv",
        "generalization": TABLE_ROOT / "generalization_compact.csv",
        "cross_model_transfer": TABLE_ROOT / "C1_cross_model_transfer.csv",
        "transfer_efficiency_exported": TABLE_ROOT / "C2_transfer_efficiency.csv",
        "judge_coverage": TABLE_ROOT / "D2_judge_coverage.csv",
        "b1_algorithm_ablation": ARTIFACTS
        / "revision_tables"
        / "table_b1_algorithm_ablation.csv",
        "b2_reward_ablation": ARTIFACTS
        / "revision_tables"
        / "table_b2_reward_ablation.csv",
        "b3_component_ablation": ARTIFACTS
        / "revision_tables"
        / "table_b3_component_ablation.csv",
    }
    result: dict[str, list[dict[str, str]]] = {}
    for name, path in sources.items():
        result[name] = read_csv(path)
        target = TABLES / path.name
        target.write_bytes(path.read_bytes())
    return result


def metric_path(method: str, target: str) -> Path:
    if method == "GPTFuzzer":
        return (
            ARTIFACTS
            / "transfer/jailbreakhub/baselines/gptfuzzer"
            / "transfer_jailbreakhub_gptfuzzer/models"
            / target
            / "25/run_000/metrics.json"
        )
    if method == "RL-based Multi-agent Framework (Ours)":
        return (
            ARTIFACTS
            / "transfer/jailbreakhub/sac/models"
            / target
            / "25/run_000/metrics.json"
        )
    slug = method.lower()
    return (
        ARTIFACTS
        / "transfer/jailbreakhub/baselines"
        / slug
        / "models"
        / target
        / "25/run_000/metrics.json"
    )


def corrected_transfer_efficiency() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        vals = []
        sources = []
        for target, _ in CORE_TARGETS:
            path = metric_path(method, target)
            data = json.loads(path.read_text(encoding="utf-8"))
            vals.append(float(data["qps"]))
            sources.append(path.relative_to(ROOT).as_posix())
        rows.append(
            {
                "Method": method,
                "Mean": round(statistics.mean(vals), 6),
                "Median": round(statistics.median(vals), 6),
                "Worst": round(max(vals), 6),
                "Targets": [display for _, display in CORE_TARGETS],
                "Values": vals,
                "Sources": sources,
            }
        )
    return rows


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 7.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIG / f"{stem}.png")
    fig.savefig(FIG / f"{stem}.pdf")
    plt.close(fig)


def figure_main_asr(rows: list[dict[str, str]]) -> None:
    targets = ["Qwen2.5-14B", "DeepSeek-R1-7B", "Vicuna-13B", "GPT-OSS-20B"]
    by_method = {row["Method"]: row for row in rows}
    x = np.arange(len(targets))
    width = 0.095
    fig, ax = plt.subplots(figsize=(10.2, 4.5))
    for idx, method in enumerate(METHODS):
        values = [float(by_method[method][f"{target} ASR"]) for target in targets]
        offset = (idx - (len(METHODS) - 1) / 2) * width
        bars = ax.bar(
            x + offset,
            values,
            width,
            label=SHORT.get(method, method),
            color=COLORS[method],
            edgecolor="black" if method.endswith("(Ours)") else "none",
            linewidth=0.8,
        )
        if method.endswith("(Ours)"):
            ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=7, fontweight="bold")
    ax.set_ylabel("Attack Success Rate")
    ax.set_xticks(x, targets)
    ax.set_ylim(0, 1.12)
    ax.grid(axis="y", alpha=0.22)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.19), frameon=False)
    ax.set_title("Fixed-budget jailbreak success across target models ($Q=25$)")
    fig.tight_layout()
    save(fig, "fig1_main_asr")


def figure_guard_asr(rows: list[dict[str, str]]) -> None:
    targets = ["Llama-Guard3-8B", "Granite-Guardian-8B", "ShieldGemma-9B"]
    matrix = np.array(
        [[float(row[f"{target} ASR"]) for target in targets] for row in rows]
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    image = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(targets)), targets)
    ax.set_yticks(range(len(rows)), [SHORT.get(r["Method"], r["Method"]) for r in rows])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value >= 0.65 else "black",
                fontweight="bold" if rows[i]["Method"].endswith("(Ours)") else "normal",
                fontsize=8,
            )
    fig.colorbar(image, ax=ax, label="Attack Success Rate", fraction=0.035, pad=0.03)
    ax.set_title("Jailbreak ASR across safety models ($Q=25$)")
    fig.tight_layout()
    save(fig, "fig2_guard_asr_heatmap")


def figure_transfer_qs(rows: list[dict[str, Any]]) -> None:
    ordered = sorted(rows, key=lambda row: float(row["Mean"]), reverse=True)
    labels = [SHORT.get(str(row["Method"]), str(row["Method"])) for row in ordered]
    values = [float(row["Mean"]) for row in ordered]
    colors = [COLORS[str(row["Method"])] for row in ordered]
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    bars = ax.barh(labels, values, color=colors)
    ax.set_xscale("log")
    ax.set_xlabel("Mean queries per success (log scale; lower is better)")
    ax.grid(axis="x", alpha=0.22, which="both")
    for bar, value in zip(bars, values):
        ax.text(value * 1.06, bar.get_y() + bar.get_height() / 2, f"{value:.1f}", va="center")
    ax.set_title("Query efficiency under JailbreakHub transfer ($Q=25$)")
    fig.tight_layout()
    save(fig, "fig3_transfer_query_efficiency")


def figure_b3(rows: list[dict[str, str]]) -> None:
    labels = ["Full", "– Rewriter", "– Stratified replay", "– Early-exit/de-dup"]
    asr = [float(row["ASR"]) for row in rows]
    qps = [float(row["Q/S"]) for row in rows]
    colors = ["#d62728", "#6c8ebf", "#82b366", "#9673a6"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.0, 3.9))
    b1 = left.bar(labels, asr, color=colors)
    left.set_ylabel("Attack Success Rate")
    left.set_ylim(0, 1.08)
    left.grid(axis="y", alpha=0.22)
    left.bar_label(b1, fmt="%.3f", padding=2, fontsize=8)
    b2 = right.bar(labels, qps, color=colors)
    right.set_yscale("log")
    right.set_ylabel("Queries per success (log; lower is better)")
    right.grid(axis="y", alpha=0.22, which="both")
    right.bar_label(b2, fmt="%.1f", padding=2, fontsize=8)
    for ax in (left, right):
        ax.tick_params(axis="x", rotation=24)
    fig.suptitle("Component ablation: effectiveness and query cost ($Q=25$)")
    fig.tight_layout()
    save(fig, "fig4_component_ablation")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    style()

    tables = table_exports()
    corrected_qs = corrected_transfer_efficiency()
    raw = raw_metrics()
    payload = {
        "schema_version": "1.0",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_scope": {
            "query_budget": 25,
            "core_targets": [display for _, display in CORE_TARGETS],
            "notes": [
                "Raw metric artifacts are embedded with paths and SHA-256 hashes.",
                "Corrected transfer efficiency uses identical four-target coverage for every method.",
                "Mistral optional expansion is retained in raw artifacts but excluded from four-target aggregates until complete.",
            ],
        },
        "tables": tables,
        "corrected_transfer_efficiency_four_targets": corrected_qs,
        "raw_metric_artifacts": raw,
    }
    result_path = OUT / "jailbreak_results_q25.json"
    write_json(result_path, payload)
    (OUT / "SHA256SUMS").write_text(
        f"{sha256(result_path)}  {result_path.name}\n", encoding="utf-8"
    )

    figure_main_asr(tables["main_5d"])
    figure_guard_asr(tables["guard_models"])
    figure_transfer_qs(corrected_qs)
    figure_b3(tables["b3_component_ablation"])

    captions = """# Paper figure captions

## Figure 1 — Main target-model ASR

**Caption.** Attack success rate across four target LLMs under a fixed query
budget of $Q=25$. All methods use the same evaluation subset, judge, and
success-based early stopping. The proposed framework reaches ASR 1.00 on
Qwen2.5-14B, DeepSeek-R1-7B, and Vicuna-13B, while its lower GPT-OSS-20B result
shows that effectiveness remains target-family dependent.

## Figure 2 — Guard-model robustness

**Caption.** Attack success rate across Llama-Guard3-8B,
Granite-Guardian-8B, and ShieldGemma-9B at fixed $Q=25$. The heat map exposes
method-specific variation across safety-model families rather than implying a
single method dominates every guard.

## Figure 3 — Transfer query efficiency

**Caption.** Mean queries per successful attack under transfer from
SORRY-Bench to unseen JailbreakHub prompts. Statistics use the same four target
models for every method; the logarithmic axis is used because query costs span
more than one order of magnitude. Lower is better. The proposed framework has
the lowest mean Q/S (5.34).

## Figure 4 — Component ablation

**Caption.** Component ablation of the proposed framework on GPT-OSS-20B at
fixed $Q=25$. The left panel reports ASR and the right panel reports Q/S on a
log scale. Removing the rewriter reduces both success and diversity, while
removing early-exit/de-duplication sharply increases query cost. Stratified
replay exhibits a metric-specific trade-off and is not claimed to improve every
reported metric.
"""
    (OUT / "FIGURE_CAPTIONS.md").write_text(captions, encoding="utf-8")
    readme = f"""# Clean Q=25 jailbreak submission bundle

Generated: {payload['generated_utc']}

- `jailbreak_results_q25.json`: consolidated table data and {len(raw)} raw
  metric artifacts with provenance hashes.
- `tables/`: copies of the finalized CSV exports.
- `figures/`: four figures in PNG and vector PDF format.
- `FIGURE_CAPTIONS.md`: publication-ready captions and interpretation.
- `SHA256SUMS`: integrity hash for the consolidated JSON.

This bundle is additive and non-destructive. Original configs, logs, events,
and metric files remain in their canonical artifact locations.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    print(result_path)
    print(f"raw_metrics={len(raw)}")
    print(f"figures={len(list(FIG.glob('*.png')))} PNG + {len(list(FIG.glob('*.pdf')))} PDF")


if __name__ == "__main__":
    main()
