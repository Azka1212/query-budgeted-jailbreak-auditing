#!/usr/bin/env python3
"""Build a non-canonical snapshot of all paper tables from current Q=25 results."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "artifacts" / "revision_tables" / "paper_tables" / "doc"
OUT = ROOT / "artifacts" / "revision_tables" / "provisional_q25"


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def row(values: list[str]) -> str:
    return " & ".join(values) + r" \\"


def simple_table(
    filename: str,
    caption: str,
    label: str,
    fields: list[str],
    rows: list[list[str]],
) -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        f"\\caption{{{caption} Values marked \\textit{{Pending}} are not yet available.}}",
        f"\\label{{{label}}}",
        "\\begin{tabular}{" + "l" + "c" * (len(fields) - 1) + "}",
        r"\toprule",
        row(fields),
        r"\midrule",
    ]
    lines.extend(row(values) for values in rows)
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    write(OUT / filename, "\n".join(lines))


def b1() -> None:
    path = ROOT / "artifacts" / "ablations" / "B1_algorithm" / "b1_algorithm_table.csv"
    records = {r["variant"]: r for r in csv.DictReader(path.open(encoding="utf-8"))}

    def value(variant: str, key: str) -> str:
        raw = records.get(variant, {}).get(key, "")
        return f"{float(raw):.3f}" if raw else r"\textit{Pending}"

    rows = [
        ["Bandit-Triad", value("bandit_triad", "asr_mean"), value("bandit_triad", "stealth_mean"),
         value("bandit_triad", "impact_mean"), value("bandit_triad", "q_per_success_mean"),
         value("bandit_triad", "distinct_2_mean")],
        ["PPO-Triad", r"\textit{Pending}", r"\textit{Pending}", r"\textit{Pending}",
         r"\textit{Pending}", r"\textit{Pending}"],
        ["Off-policy RL (Full)", value("sac_triad", "asr_mean"), value("sac_triad", "stealth_mean"),
         value("sac_triad", "impact_mean"), value("sac_triad", "q_per_success_mean"),
         value("sac_triad", "distinct_2_mean")],
    ]
    simple_table(
        "B1_algorithm_ablation.tex",
        r"\textbf{Algorithm Ablation (provisional, fixed $Q=25$).}",
        "tab:b1_algorithm_ablation",
        ["Variant", "ASR", "Stealth", "Impact", "Q/S", "D-2"],
        rows,
    )


def b2() -> None:
    pending = r"\textit{Pending}"
    simple_table(
        "B2_reward_ablation.tex",
        r"\textbf{Reward Ablation (provisional, fixed $Q=25$).}",
        "tab:b2_reward_ablation",
        ["Reward Variant", "ASR", "Q/S", "D-1", "D-2 / D-3"],
        [
            ["Binary reward", pending, pending, pending, pending],
            ["Single scalar", pending, pending, pending, pending],
            ["Full multi-dimensional curriculum", pending, pending, pending, pending],
        ],
    )


def b3() -> None:
    base = ROOT / "artifacts" / "ablations" / "b3" / "runs"
    labels = [
        ("full", "RL-based Multi-agent Framework"),
        ("minus_rewriter", "-- Rewriter"),
        ("minus_stratified_replay", "-- Stratified Replay"),
        ("minus_earlyexit_dedup", "-- Early-Exit / De-Dup"),
    ]
    rows: list[list[str]] = []
    for slug, label in labels:
        path = base / slug / "seed_1" / "metrics.json"
        if not path.exists():
            rows.append([label] + [r"\textit{Pending}"] * 4)
            continue
        metrics = json.loads(path.read_text(encoding="utf-8"))["eval"]
        rows.append([
            label,
            f"{metrics['asr']:.3f}",
            f"{metrics['qs']:.3f}",
            f"{metrics['stealth']:.3f}",
            f"{metrics['dist2']:.3f}",
        ])
    simple_table(
        "B3_component_ablation.tex",
        r"\textbf{Component Ablation (provisional, fixed $Q=25$).}",
        "tab:b3_component_ablation",
        ["Variant", "ASR", "Q/S", "Stealth", "D-2"],
        rows,
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    copies = {
        "main_5d_results.tex": "main_5d_results.tex",
        "GUARD-models.tex": "GUARD-models.tex",
        "C1_cross_model_transfer.tex": "C1_cross_model_transfer.tex",
        "C2_transfer_efficiency.tex": "C2_transfer_efficiency.tex",
        "cross.tex": "cross_dataset_transfer.tex",
        "gen.tex": "generalization_compact.tex",
        "D2_judge_coverage.tex": "D2_judge_coverage.tex",
    }
    for source, destination in copies.items():
        shutil.copy2(SOURCE / source, OUT / destination)

    b1()
    b2()
    b3()

    statuses = [
        ("Main 5D results", "main_5d_results.tex", "Partial", "Baselines and one-step SAC available; fair multi-step SAC pending"),
        ("Guard models", "GUARD-models.tex", "Partial", "Baselines and one-step SAC available; fair multi-step SAC pending"),
        ("B1 Algorithm ablation", "B1_algorithm_ablation.tex", "Running", "PPO corrective run pending"),
        ("B2 Reward ablation", "B2_reward_ablation.tex", "Waiting", "Three variants pending"),
        ("B3 Component ablation", "B3_component_ablation.tex", "Running", "3/4 variants available"),
        ("C1 Cross-model transfer", "C1_cross_model_transfer.tex", "Partial", "SAC complete; Mistral baseline expansion running"),
        ("C2 Transfer efficiency", "C2_transfer_efficiency.tex", "Provisional", "Computed from currently completed transfer rows"),
        ("Cross-dataset transfer", "cross_dataset_transfer.tex", "Provisional", "Computed from currently completed Q=25 rows"),
        ("Generalization compact", "generalization_compact.tex", "Available", "Includes dataset and model transfer"),
        ("D2 Judge coverage", "D2_judge_coverage.tex", "Available", "Fresh 25-item Q=25 diagnostic subset"),
    ]
    lines = [
        "# Provisional Q=25 paper tables",
        "",
        "Snapshot generated from currently validated artifacts. `Pending` means the experiment has not produced a result yet.",
        "",
        "| # | Table | File | Status | Remaining work |",
        "|---:|---|---|---|---|",
    ]
    for index, (name, filename, status, note) in enumerate(statuses, 1):
        lines.append(f"| {index} | {name} | [{filename}]({filename}) | {status} | {note} |")
    write(OUT / "README.md", "\n".join(lines))
    print(f"[provisional] wrote {len(statuses)} tables to {OUT}")


if __name__ == "__main__":
    main()
