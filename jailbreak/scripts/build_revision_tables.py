#!/usr/bin/env python3
"""Collect rerun metrics into revision-ready paper tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = REPO_ROOT / "artifacts" / "revision_tables"

TARGET_IDS = {
    "mistral_instruct": "ollama/mistral:instruct",
    "qwen2_5_14b": "ollama/qwen2.5:14b-instruct",
    "deepseek_r1_7b": "ollama/deepseek-r1:7b",
    "vicuna_13b": "ollama/vicuna:13b",
    "gpt_oss_20b": "ollama/gpt-oss:20b",
    "llama_guard3_8b": "ollama/llama-guard3:8b",
    "granite_guardian_8b": "ollama/granite3-guardian:8b",
    "shieldgemma_9b": "ollama/shieldgemma:9b",
}


def _slug_model_id(model_id: str) -> str:
    import re
    return re.sub(r"[^a-z0-9._-]+", "_", model_id.replace("/", "_").replace(":", "_").strip().lower()).strip("_")


SLUG_TO_TARGET = {_slug_model_id(model_id): short for short, model_id in TARGET_IDS.items()}

PAPER_ROOT = OUT_ROOT / "paper_tables"
DOC_ROOT = PAPER_ROOT / "doc"

DISPLAY_METHODS = {
    "gcg": "GCG",
    "autodan": "AutoDAN",
    "gptfuzzer": "GPTFuzzer",
    "tap": "TAP",
    "pair": "PAIR",
    "rlbreaker": "RLBreaker",
    "xjailbreak": "xJailbreak",
    "SAC-Triad": "RL-based Multi-agent Framework (Ours)",
}
METHOD_ORDER = ["gcg", "autodan", "gptfuzzer", "pair", "tap", "rlbreaker", "xjailbreak", "SAC-Triad"]
STANDARD_TARGETS = ["mistral_instruct", "deepseek_r1_7b", "qwen2_5_14b", "gpt_oss_20b", "vicuna_13b"]
GUARD_TARGETS = ["llama_guard3_8b", "granite_guardian_8b", "shieldgemma_9b"]
TARGET_DISPLAY = {
    "mistral_instruct": "Mistral-7B",
    "qwen2_5_14b": "Qwen2.5-14B",
    "deepseek_r1_7b": "DeepSeek-R1-7B",
    "vicuna_13b": "Vicuna-13B",
    "gpt_oss_20b": "GPT-OSS-20B",
    "llama_guard3_8b": "Llama-Guard3-8B",
    "granite_guardian_8b": "Granite-Guardian-8B",
    "shieldgemma_9b": "ShieldGemma-9B",
}
METRIC_ORDER = ["ASR", "Stealth", "Novelty", "Impact", "Q/S"]



def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _num(value: Any, default: float = float("nan")) -> float:
    if value in (None, "", "NA", "nan"):
        return default
    try:
        return float(value)
    except Exception:
        return default


def _fmt(value: Any, digits: int = 3) -> str:
    x = _num(value)
    if math.isnan(x):
        return ""
    if math.isinf(x):
        return "inf"
    return f"{x:.{digits}f}"


def _pick(d: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in d:
            return d[key]
    lower = {str(k).lower(): v for k, v in d.items()}
    for key in keys:
        if key.lower() in lower:
            return lower[key.lower()]
    return default


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _write_md(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("| " + " | ".join(fields) + " |")
    lines.append("| " + " | ".join("---" for _ in fields) + " |")
    for row in rows:
        vals = [str(row.get(field, "")) for field in fields]
        lines.append("| " + " | ".join(vals) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _method_from_path(path: Path) -> str:
    parts = path.parts
    try:
        idx = parts.index("baselines")
        return parts[idx + 1]
    except Exception:
        return "unknown"


def _normalize_baseline_metric(path: Path, data: dict[str, Any]) -> dict[str, Any] | None:
    target = _pick(data, "target_short_name", "target_short")
    budget = _pick(data, "query_budget", "budget")
    if target is None or budget is None:
        return None
    return {
        "family": "baseline",
        "method": _method_from_path(path),
        "target": str(target),
        "budget": int(_num(budget, -1)),
        "run": str(_pick(data, "repeat_idx", "run", default="")),
        "asr": _num(_pick(data, "asr", "ASR")),
        "qps": _num(_pick(data, "qps", "q_per_success", "QueriesPerSuccess")),
        "success": _num(_pick(data, "avg_success", "success")),
        "stealth": _num(_pick(data, "avg_stealth", "stealth")),
        "novelty": _num(_pick(data, "avg_novelty", "novelty")),
        "efficiency": _num(_pick(data, "avg_efficiency", "efficiency")),
        "impact": _num(_pick(data, "avg_impact", "impact")),
        "distinct_2": _num(_pick(data, "distinct_2", "Distinct2", "dist2")),
        "num_pairs": _num(_pick(data, "num_pairs", "pairs")),
        "num_success": _num(_pick(data, "num_success", "successes")),
        "path": path.as_posix(),
    }


def collect_baselines(budget: int, methods: set[str], targets: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    base = REPO_ROOT / "artifacts" / "baselines"
    for path in base.glob("**/metrics.json"):
        data = _read_json(path)
        if not data:
            continue
        row = _normalize_baseline_metric(path, data)
        if not row:
            continue
        if row["budget"] != budget:
            continue
        if methods and row["method"] not in methods:
            continue
        if targets and row["target"] not in targets:
            continue
        rows.append(row)
    return rows


def _normalize_minimal_sac_metric(path: Path, data: dict[str, Any], budget: int) -> dict[str, Any] | None:
    # Fresh `python -m src.cli test-all` writes artifacts/eval/<run_slug>/metrics_OOD-JBB.json.
    # The JSON is minimal, so infer target from the parent run slug and normalize to table schema.
    target = SLUG_TO_TARGET.get(path.parent.name)
    if target is None:
        return None

    counts = data.get("counts", {}) if isinstance(data.get("counts"), dict) else {}
    metrics = data.get("metrics", {}) if isinstance(data.get("metrics"), dict) else {}
    judge_means = data.get("judge_means", {}) if isinstance(data.get("judge_means"), dict) else {}
    qps = _pick(metrics, "QueriesPerSuccess", "qps", "q_per_success")
    return {
        "family": "main_sac",
        "method": "SAC-Triad",
        "target": target,
        "budget": int(budget),
        "run": "fresh_test_all",
        "asr": _num(_pick(metrics, "ASR", "asr")),
        "qps": _num(qps),
        "success": _num(_pick(judge_means, "Success", "success")),
        "stealth": _num(_pick(judge_means, "Stealth", "stealth")),
        "novelty": _num(_pick(judge_means, "Novelty", "novelty")),
        "efficiency": _num(_pick(judge_means, "Efficiency", "efficiency")),
        "impact": _num(_pick(judge_means, "Impact", "impact")),
        "distinct_2": _num(_pick(metrics, "Distinct2", "distinct_2", "dist2")),
        "num_pairs": _num(_pick(counts, "pairs", "num_pairs")),
        "num_success": _num(_pick(counts, "successes", "num_success")),
        "path": path.as_posix(),
    }


def collect_sac_eval(budget: int, targets: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    base = REPO_ROOT / "artifacts" / "eval"

    # Legacy checkpoint evaluator writes baseline-style metrics.json.
    for path in base.glob("**/metrics.json"):
        data = _read_json(path)
        if not data:
            continue
        row = _normalize_baseline_metric(path, data)
        if not row:
            continue
        row["family"] = "sac_eval"
        row["method"] = "SAC-Triad"
        if row["budget"] != budget:
            continue
        if targets and row["target"] not in targets:
            continue
        rows.append(row)

    # Fresh main_sac test-all writes minimal metrics_OOD-JBB.json.
    for path in base.glob("*/metrics_OOD-JBB.json"):
        data = _read_json(path)
        if not data:
            continue
        row = _normalize_minimal_sac_metric(path, data, budget)
        if not row:
            continue
        if targets and row["target"] not in targets:
            continue
        rows.append(row)

    return rows


def _aggregate(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["method"]), str(row["target"]), int(row["budget"]))].append(row)

    out: list[dict[str, Any]] = []
    for (method, target, budget), items in sorted(grouped.items()):
        def vals(key: str) -> list[float]:
            return [float(r[key]) for r in items if not math.isnan(_num(r.get(key)))]
        def avg(key: str) -> float:
            v = vals(key)
            return mean(v) if v else float("nan")
        def sd(key: str) -> float:
            v = vals(key)
            return stdev(v) if len(v) > 1 else 0.0
        out.append({
            "Method": method,
            "Target": target,
            "Q": budget,
            "Runs": len(items),
            "ASR": _fmt(avg("asr")),
            "ASR_std": _fmt(sd("asr")),
            "Q/S": _fmt(avg("qps")),
            "Success": _fmt(avg("success")),
            "Stealth": _fmt(avg("stealth")),
            "Novelty": _fmt(avg("novelty")),
            "Efficiency": _fmt(avg("efficiency")),
            "Impact": _fmt(avg("impact")),
            "Distinct-2": _fmt(avg("distinct_2")),
        })
    return out


def collect_b1() -> list[dict[str, Any]]:
    path = REPO_ROOT / "artifacts" / "ablations" / "B1_algorithm" / "b1_algorithm_table.csv"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))




def collect_b2() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    base = REPO_ROOT / "artifacts" / "ablations" / "b2"
    for path in base.glob("*/**/eval_metrics.json"):
        data = _read_json(path)
        if not data:
            continue
        parts = path.relative_to(base).parts
        variant = parts[0] if parts else "unknown"
        target = parts[1] if len(parts) > 1 else ""
        seed = parts[2].replace("seed_", "") if len(parts) > 2 else ""
        rows.append({
            "Variant": variant,
            "Target": target,
            "Seed": seed,
            "ASR": _fmt(_pick(data, "asr")),
            "Q/S": _fmt(_pick(data, "q_per_success", "qps")),
            "Stealth": _fmt(_pick(data, "stealth")),
            "Impact": _fmt(_pick(data, "impact")),
            "Distinct-1": _fmt(_pick(data, "distinct_1")),
            "Distinct-2": _fmt(_pick(data, "distinct_2")),
            "Distinct-3": _fmt(_pick(data, "distinct_3")),
            "path": path.as_posix(),
        })
    return sorted(rows, key=lambda r: (r["Variant"], r["Target"], r["Seed"]))


def collect_b3() -> list[dict[str, Any]]:
    path = REPO_ROOT / "artifacts" / "ablations" / "b3" / "tables" / "table_b3_component_ablation.csv"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))



def _display_method(method: str) -> str:
    return DISPLAY_METHODS.get(method, method)


def _method_sort_key(method: str) -> int:
    try:
        return METHOD_ORDER.index(method)
    except ValueError:
        return len(METHOD_ORDER)


def _rows_to_lookup(table_rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(str(r["Method"]), str(r["Target"])): r for r in table_rows}


def _write_latex_fragment(path: Path, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body.rstrip() + "\n", encoding="utf-8")


def _wide_rows(table_rows: list[dict[str, Any]], targets: list[str]) -> tuple[list[dict[str, Any]], list[str]]:
    lookup = _rows_to_lookup(table_rows)
    methods = sorted({str(r["Method"]) for r in table_rows}, key=_method_sort_key)
    fields = ["Method"]
    for target in targets:
        for metric in METRIC_ORDER:
            fields.append(f"{TARGET_DISPLAY[target]} {metric}")
    rows: list[dict[str, Any]] = []
    for method in methods:
        row: dict[str, Any] = {"Method": _display_method(method)}
        for target in targets:
            src = lookup.get((method, target), {})
            for metric in METRIC_ORDER:
                row[f"{TARGET_DISPLAY[target]} {metric}"] = src.get(metric, "")
        rows.append(row)
    return rows, fields


def _latex_row(values: list[str]) -> str:
    return " & ".join(values) + r" \\" 


def _latex_wide_table(table_rows: list[dict[str, Any]], targets: list[str], *, caption: str, label: str) -> str:
    lookup = _rows_to_lookup(table_rows)
    methods = sorted({str(r["Method"]) for r in table_rows}, key=_method_sort_key)
    colspec = "l " + " ".join("ccccc" for _ in targets)
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\resizebox{\textwidth}{!}{%",
        f"\\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
    ]
    header = [r"\multirow{2}{*}{\textbf{Method}}"]
    for target in targets:
        header.append(f"\\multicolumn{{5}}{{c}}{{\\textbf{{{TARGET_DISPLAY[target]}}}}}")
    lines.append(_latex_row(header))
    cmid = []
    start_col = 2
    for _target in targets:
        cmid.append(f"\\cmidrule(lr){{{start_col}-{start_col + 4}}}")
        start_col += 5
    lines.append("".join(cmid))
    metric_head = [""] + [m for _target in targets for m in METRIC_ORDER]
    lines.append(_latex_row(metric_head))
    lines.append(r"\midrule")
    for method in methods:
        vals = [f"\\textbf{{{_display_method(method)}}}" if method == "SAC-Triad" else _display_method(method)]
        for target in targets:
            src = lookup.get((method, target), {})
            for metric in METRIC_ORDER:
                vals.append(str(src.get(metric, "--") or "--"))
        lines.append(_latex_row(vals))
        if method == "xjailbreak":
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}", "}", r"\end{table*}"])
    return "\n".join(lines)


def write_paper_main_and_guard_tables(table_rows: list[dict[str, Any]]) -> None:
    main_rows, main_fields = _wide_rows(table_rows, STANDARD_TARGETS)
    _write_csv(PAPER_ROOT / "main_5d_results_wide.csv", main_rows, main_fields)
    _write_md(PAPER_ROOT / "main_5d_results_wide.md", main_rows, main_fields)
    _write_latex_fragment(
        DOC_ROOT / "main_5d_results.tex",
        _latex_wide_table(
            table_rows,
            STANDARD_TARGETS,
            caption="\\textbf{Multi-dimensional jailbreak performance across target models}. All methods use fixed query budget $Q=25$.",
            label="tab:main_5d_results",
        ),
    )

    guard_rows, guard_fields = _wide_rows(table_rows, GUARD_TARGETS)
    _write_csv(PAPER_ROOT / "GUARD-models_wide.csv", guard_rows, guard_fields)
    _write_md(PAPER_ROOT / "GUARD-models_wide.md", guard_rows, guard_fields)
    _write_latex_fragment(
        DOC_ROOT / "GUARD-models.tex",
        _latex_wide_table(
            table_rows,
            GUARD_TARGETS,
            caption="\\textbf{Jailbreak performance across safety (guard) models}. All methods use fixed query budget $Q=25$.",
            label="tab:safety_models_full_table_star",
        ),
    )


def _simple_latex_table(path: Path, rows: list[dict[str, Any]], fields: list[str], *, caption: str, label: str) -> None:
    colspec = "l" + "c" * max(0, len(fields) - 1)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\scriptsize",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
        _latex_row(fields),
        r"\midrule",
    ]
    for row in rows:
        lines.append(_latex_row([str(row.get(f, "")) for f in fields]))
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    _write_latex_fragment(path, "\n".join(lines))


def write_paper_ablation_tables(b1_rows: list[dict[str, Any]], b2_rows: list[dict[str, Any]], b3_rows: list[dict[str, Any]]) -> None:
    if b1_rows:
        fields = list(b1_rows[0])
        _simple_latex_table(DOC_ROOT / "B1.tex", b1_rows, fields, caption="\\textbf{Algorithm Ablation (RL Engine Comparison).} Fixed $Q=25$.", label="tab:b1_algorithm_ablation")
    if b2_rows:
        fields = [f for f in ["Variant", "ASR", "Q/S", "Distinct-1", "Distinct-2", "Distinct-3"] if f in b2_rows[0]]
        _simple_latex_table(DOC_ROOT / "B2.tex", b2_rows, fields, caption="\\textbf{Reward Ablation (Effect of Reward Shaping).} Fixed $Q=25$.", label="tab:b2_reward_ablation")
    if b3_rows:
        fields = list(b3_rows[0])
        _simple_latex_table(DOC_ROOT / "B3.tex", b3_rows, fields, caption="\\textbf{Component Ablation (Architecture and Stability).} Fixed $Q=25$.", label="tab:b3_component_ablation")



def collect_transfer_metrics(budget: int, methods: set[str], targets: set[str], dataset: str = "jailbreakhub") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    base = REPO_ROOT / "artifacts" / "transfer" / dataset
    for path in (base / "baselines").glob("*/models/*/*/run_*/metrics.json"):
        data = _read_json(path)
        if not data:
            continue
        row = _normalize_baseline_metric(path, data)
        if not row:
            continue
        # path: baselines/<method>/models/<target>/<budget>/run_x/metrics.json
        try:
            method = path.relative_to(base / "baselines").parts[0]
        except Exception:
            method = row["method"]
        row["method"] = method
        row["family"] = f"transfer:{dataset}"
        row["transfer_dataset"] = dataset
        if row["budget"] != budget:
            continue
        if methods and method not in methods:
            continue
        if targets and row["target"] not in targets:
            continue
        rows.append(row)
    for path in (base / "sac").glob("models/*/*/run_*/metrics.json"):
        data = _read_json(path)
        if not data:
            continue
        row = _normalize_baseline_metric(path, data)
        if not row:
            continue
        row["family"] = f"transfer:{dataset}"
        row["method"] = "SAC-Triad"
        row["transfer_dataset"] = dataset
        if row["budget"] != budget:
            continue
        if targets and row["target"] not in targets:
            continue
        rows.append(row)
    return rows


def _avg_rows(rows: list[dict[str, Any]], method: str, keys: list[str]) -> dict[str, str]:
    subset = [r for r in rows if r.get("method") == method]
    out: dict[str, str] = {"Method": _display_method(method)}
    for key in keys:
        vals = [_num(r.get(key)) for r in subset]
        vals = [v for v in vals if not math.isnan(v) and not math.isinf(v)]
        out[key] = _fmt(mean(vals)) if vals else ""
    return out


def write_transfer_tables(main_rows: list[dict[str, Any]], transfer_rows: list[dict[str, Any]], budget: int) -> None:
    if not transfer_rows:
        return
    methods = sorted({str(r["method"]) for r in transfer_rows}, key=_method_sort_key)

    # cross.tex / C1 dataset transfer: compare averaged JBB vs JailbreakHub by method.
    c_rows: list[dict[str, Any]] = []
    for method in methods:
        jbb = [r for r in main_rows if r.get("method") == method and r.get("target") in STANDARD_TARGETS]
        hub = [r for r in transfer_rows if r.get("method") == method and r.get("target") in STANDARD_TARGETS]
        for label, subset in [("JBB-OOD", jbb), ("JailbreakHub", hub)]:
            row = {"Method": _display_method(method), "TestDataset": label}
            for out_key, raw_key in [("ASR", "asr"), ("Stealth", "stealth"), ("Q/S", "qps"), ("D-2", "distinct_2")]:
                vals = [_num(r.get(raw_key)) for r in subset]
                vals = [v for v in vals if not math.isnan(v) and not math.isinf(v)]
                row[out_key] = _fmt(mean(vals)) if vals else ""
            c_rows.append(row)
    c_fields = ["Method", "TestDataset", "ASR", "Stealth", "Q/S", "D-2"]
    _write_csv(PAPER_ROOT / "cross_dataset_transfer.csv", c_rows, c_fields)
    _write_md(PAPER_ROOT / "cross_dataset_transfer.md", c_rows, c_fields)
    _simple_latex_table(DOC_ROOT / "cross.tex", c_rows, c_fields, caption=f"\\textbf{{{{Cross-dataset transfer at fixed $Q={budget}$.}}}}", label="tab:c1_dataset_transfer")

    # C1 target transfer ASR matrix on transfer dataset.
    matrix_rows: list[dict[str, Any]] = []
    for method in methods:
        row = {"Method": _display_method(method)}
        for target in STANDARD_TARGETS:
            vals = [_num(r.get("asr")) for r in transfer_rows if r.get("method") == method and r.get("target") == target]
            vals = [v for v in vals if not math.isnan(v)]
            row[TARGET_DISPLAY[target]] = _fmt(mean(vals)) if vals else ""
        matrix_rows.append(row)
    matrix_fields = ["Method"] + [TARGET_DISPLAY[t] for t in STANDARD_TARGETS]
    _write_csv(PAPER_ROOT / "C1_cross_model_transfer.csv", matrix_rows, matrix_fields)
    _write_md(PAPER_ROOT / "C1_cross_model_transfer.md", matrix_rows, matrix_fields)
    _simple_latex_table(DOC_ROOT / "C1_cross_model_transfer.tex", matrix_rows, matrix_fields, caption=f"\\textbf{{{{Cross-model transfer ASR at fixed $Q={budget}$.}}}}", label="tab:C1_transfer_asr")

    # C2 Q/S summary across transfer targets.
    qs_rows: list[dict[str, Any]] = []
    for method in methods:
        vals = [_num(r.get("qps")) for r in transfer_rows if r.get("method") == method]
        vals = [v for v in vals if not math.isnan(v) and not math.isinf(v)]
        qs_rows.append({
            "Method": _display_method(method),
            "Mean": _fmt(mean(vals)) if vals else "",
            "Median": _fmt(sorted(vals)[len(vals)//2]) if vals else "",
            "Worst": _fmt(max(vals)) if vals else "",
        })
    qs_fields = ["Method", "Mean", "Median", "Worst"]
    _write_csv(PAPER_ROOT / "C2_transfer_efficiency.csv", qs_rows, qs_fields)
    _write_md(PAPER_ROOT / "C2_transfer_efficiency.md", qs_rows, qs_fields)
    _simple_latex_table(DOC_ROOT / "C2_transfer_efficiency.tex", qs_rows, qs_fields, caption=f"\\textbf{{{{Query efficiency under transfer at fixed $Q={budget}$.}}}}", label="tab:C2_transfer_qs")

    # gen.tex: compact Ours-only summary, generated only from real rows.
    ours_jbb = [r for r in main_rows if r.get("method") == "SAC-Triad" and r.get("target") in STANDARD_TARGETS]
    ours_hub = [r for r in transfer_rows if r.get("method") == "SAC-Triad" and r.get("target") in STANDARD_TARGETS]
    gen_rows: list[dict[str, Any]] = []
    for label, subset in [("JBB-OOD", ours_jbb), ("JailbreakHub", ours_hub)]:
        if not subset:
            continue
        row = {"Setting": label}
        for out_key, raw_key in [("ASR", "asr"), ("Stealth", "stealth"), ("Q/S", "qps"), ("D-2", "distinct_2")]:
            vals = [_num(r.get(raw_key)) for r in subset]
            vals = [v for v in vals if not math.isnan(v) and not math.isinf(v)]
            row[out_key] = _fmt(mean(vals)) if vals else ""
        gen_rows.append(row)
    if gen_rows:
        gen_fields = ["Setting", "ASR", "Stealth", "Q/S", "D-2"]
        _write_csv(PAPER_ROOT / "generalization_compact.csv", gen_rows, gen_fields)
        _write_md(PAPER_ROOT / "generalization_compact.md", gen_rows, gen_fields)
        _simple_latex_table(DOC_ROOT / "gen.tex", gen_rows, gen_fields, caption=f"\\textbf{{{{Generalization under dataset shift at fixed $Q={budget}$.}}}}", label="tab:generalization_compact")


def write_missing_paper_tables_report() -> None:
    rows = [
        {"PaperInput": "doc/main_5d_results", "Status": "generated_after_rerun", "Output": "paper_tables/doc/main_5d_results.tex"},
        {"PaperInput": "doc/GUARD-models", "Status": "generated_after_rerun", "Output": "paper_tables/doc/GUARD-models.tex"},
        {"PaperInput": "doc/B1", "Status": "generated_after_rerun", "Output": "paper_tables/doc/B1.tex"},
        {"PaperInput": "doc/B2", "Status": "generated_after_rerun", "Output": "paper_tables/doc/B2.tex"},
        {"PaperInput": "doc/B3", "Status": "generated_after_rerun", "Output": "paper_tables/doc/B3.tex"},
        {"PaperInput": "doc/gen", "Status": "generated_after_transfer_rerun", "Output": "paper_tables/doc/gen.tex"},
        {"PaperInput": "doc/cross", "Status": "generated_after_transfer_rerun", "Output": "paper_tables/doc/cross.tex"},
        {"PaperInput": "doc/C1_cross_model_transfer", "Status": "generated_after_transfer_rerun", "Output": "paper_tables/doc/C1_cross_model_transfer.tex"},
        {"PaperInput": "doc/C2_transfer_efficiency", "Status": "generated_after_transfer_rerun", "Output": "paper_tables/doc/C2_transfer_efficiency.tex"},
        {"PaperInput": "doc/D1_human_alignment", "Status": "manual_or_archival", "Output": "needs fresh human annotation sample or archival label"},
        {"PaperInput": "doc/D2_cross_judge", "Status": "blocked_or_archival", "Output": "needs fresh alternate-judge scoring or archival label"},
    ]
    fields = ["PaperInput", "Status", "Output"]
    _write_csv(PAPER_ROOT / "paper_table_coverage.csv", rows, fields)
    _write_md(PAPER_ROOT / "paper_table_coverage.md", rows, fields)


def write_gap_report(budget: int, methods: list[str], targets: list[str], aggregate_rows: list[dict[str, Any]]) -> None:
    present = {(r["Method"], r["Target"], int(r["Q"])) for r in aggregate_rows}
    rows = []
    for method in methods + ["SAC-Triad"]:
        for target in targets:
            status = "present" if (method, target, budget) in present else "missing"
            rows.append({"Method": method, "Target": target, "Q": budget, "Status": status})
    fields = ["Method", "Target", "Q", "Status"]
    _write_csv(OUT_ROOT / f"coverage_budget_{budget}.csv", rows, fields)
    _write_md(OUT_ROOT / f"coverage_budget_{budget}.md", rows, fields)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--budget", type=int, required=True)
    ap.add_argument("--targets", nargs="*", default=[])
    ap.add_argument("--methods", nargs="*", default=[])
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    targets = set(args.targets)
    methods = set(args.methods)

    raw_rows = collect_baselines(args.budget, methods, targets) + collect_sac_eval(args.budget, targets)
    transfer_rows = collect_transfer_metrics(args.budget, methods, set(STANDARD_TARGETS), dataset="jailbreakhub")
    raw_fields = [
        "family", "method", "target", "budget", "run", "asr", "qps", "success",
        "stealth", "novelty", "efficiency", "impact", "distinct_2", "num_pairs",
        "num_success", "path",
    ]
    _write_csv(OUT_ROOT / f"raw_metrics_budget_{args.budget}.csv", raw_rows, raw_fields)

    table = _aggregate(raw_rows)
    fields = [
        "Method", "Target", "Q", "Runs", "ASR", "ASR_std", "Q/S", "Success",
        "Stealth", "Novelty", "Efficiency", "Impact", "Distinct-2",
    ]
    _write_csv(OUT_ROOT / f"table_main_results_budget_{args.budget}.csv", table, fields)
    _write_md(OUT_ROOT / f"table_main_results_budget_{args.budget}.md", table, fields)
    write_paper_main_and_guard_tables(table)

    b1_rows = collect_b1()
    if b1_rows:
        b1_fields = list(b1_rows[0])
        _write_csv(OUT_ROOT / "table_b1_algorithm_ablation.csv", b1_rows, b1_fields)
        _write_md(OUT_ROOT / "table_b1_algorithm_ablation.md", b1_rows, b1_fields)

    b2_rows = collect_b2()
    if b2_rows:
        b2_fields = list(b2_rows[0])
        _write_csv(OUT_ROOT / "table_b2_reward_ablation.csv", b2_rows, b2_fields)
        _write_md(OUT_ROOT / "table_b2_reward_ablation.md", b2_rows, b2_fields)

    b3_rows = collect_b3()
    if b3_rows:
        b3_fields = list(b3_rows[0])
        _write_csv(OUT_ROOT / "table_b3_component_ablation.csv", b3_rows, b3_fields)
        _write_md(OUT_ROOT / "table_b3_component_ablation.md", b3_rows, b3_fields)

    write_paper_ablation_tables(b1_rows, b2_rows, b3_rows)
    write_transfer_tables(raw_rows, transfer_rows, args.budget)
    write_missing_paper_tables_report()
    write_gap_report(args.budget, args.methods, args.targets, table)
    print(f"[tables] wrote revision tables under {OUT_ROOT}")
    print(f"[tables] raw metric files found: {len(raw_rows)}")
    print(f"[tables] aggregate rows: {len(table)}")
    print(f"[tables] transfer metric files found: {len(transfer_rows)}")


if __name__ == "__main__":
    main()
