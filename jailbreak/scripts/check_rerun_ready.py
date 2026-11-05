#!/usr/bin/env python3
"""Preflight checks for the fixed-budget paper rerun."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET_TAGS = {
    "qwen2_5_14b": "qwen2.5:14b-instruct",
    "deepseek_r1_7b": "deepseek-r1:7b",
    "vicuna_13b": "vicuna:13b",
    "gpt_oss_20b": "gpt-oss:20b",
    "llama_guard3_8b": "llama-guard3:8b",
    "granite_guardian_8b": "granite3-guardian:8b",
    "shieldgemma_9b": "shieldgemma:9b",
}
SUPPORT_TAGS = {"llama3:instruct", "yi:9b"}
EXPECTED_METHODS = {"gcg", "autodan", "gptfuzzer", "tap", "pair", "rlbreaker", "xjailbreak"}


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise RuntimeError(f"Expected mapping YAML: {path}")
    return data


def _ollama_tags() -> set[str]:
    try:
        out = subprocess.check_output(["ollama", "list"], text=True)
    except Exception as exc:
        print(f"WARN ollama list failed: {exc}")
        return set()
    tags = set()
    for line in out.splitlines()[1:]:
        parts = line.split()
        if parts:
            tags.add(parts[0])
    return tags


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--budget", type=int, default=25)
    ap.add_argument("--skip-ollama", action="store_true")
    args = ap.parse_args()

    errors: list[str] = []
    warnings: list[str] = []

    manifest_path = REPO_ROOT / "artifacts" / "rerun_fixed_budget" / f"budget_{args.budget}" / "rerun_manifest.json"
    if not manifest_path.exists():
        errors.append(f"missing manifest: {manifest_path}")
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("budget") != args.budget:
            errors.append(f"manifest budget is {manifest.get('budget')}, expected {args.budget}")
        targets = set(manifest.get("targets", []))
        if targets != set(TARGET_TAGS):
            errors.append(f"manifest targets mismatch: {sorted(targets)}")
        methods = set(manifest.get("methods", []))
        if methods != EXPECTED_METHODS:
            errors.append(f"manifest methods mismatch: {sorted(methods)}")
        item_names = {(i.get("group"), i.get("name")) for i in manifest.get("items", [])}
        required_items = [
            ("transfer", "export_seeds:jailbreakhub"),
            ("transfer", "jailbreakhub:gcg"),
            ("transfer", "jailbreakhub:autodan"),
            ("transfer", "jailbreakhub:gptfuzzer"),
            ("transfer", "jailbreakhub:tap"),
            ("transfer", "jailbreakhub:pair"),
            ("transfer", "jailbreakhub:rlbreaker"),
            ("transfer", "jailbreakhub:xjailbreak"),
            ("ablations", "b1_algorithm"),
            ("ablations", "b2_reward:binary"),
            ("ablations", "b2_reward:scalar"),
            ("ablations", "b2_reward:full5d"),
            ("ablations", "b3_component"),
            ("tables", "build_revision_tables"),
        ]
        for required in required_items:
            if required not in item_names:
                errors.append(f"manifest missing item: {required}")

    b1 = _load_yaml(REPO_ROOT / "artifacts" / "rerun_fixed_budget" / f"budget_{args.budget}" / "configs" / "ablations" / "b1_algorithm.yaml")
    b3 = _load_yaml(REPO_ROOT / "artifacts" / "rerun_fixed_budget" / f"budget_{args.budget}" / "configs" / "ablations" / "b3_component.yaml")
    if int(b1.get("eval_budget", -1)) != args.budget:
        errors.append("B1 eval_budget is not patched to fixed budget")
    if b1.get("target_model") != "ollama/gpt-oss:20b":
        errors.append("B1 target_model is not ollama/gpt-oss:20b")
    if int(b3.get("budget_k", -1)) != args.budget:
        errors.append("B3 budget_k is not patched to fixed budget")

    b2_root = REPO_ROOT / "artifacts" / "rerun_fixed_budget" / f"budget_{args.budget}" / "configs" / "ablations_b2"
    for name in ["reward_binary.yaml", "reward_scalar.yaml", "reward_full5d.yaml"]:
        b2 = _load_yaml(b2_root / name)
        env = b2.get("env", {}) if isinstance(b2.get("env"), dict) else {}
        eval_cfg = b2.get("eval", {}) if isinstance(b2.get("eval"), dict) else {}
        if int(env.get("max_query_budget", -1)) != args.budget:
            errors.append(f"{name} env.max_query_budget is not patched to fixed budget")
        if eval_cfg.get("query_budgets") != [args.budget]:
            errors.append(f"{name} eval.query_budgets is not patched to [{args.budget}]")
        if b2.get("judge", {}).get("model_id") == "ollama/llama3:judge":
            errors.append(f"{name} still uses stale judge alias")
        if b2.get("rewriter", {}).get("model_id") == "ollama/yi-9b":
            errors.append(f"{name} still uses stale rewriter alias")

    stack = _load_yaml(REPO_ROOT / "config" / "stack.yaml")
    app = stack.get("app", stack)
    stack_ids = {str(t.get("id")) for t in app.get("targets", []) if isinstance(t, dict) and t.get("enabled", True)}
    expected_ids = {"ollama/" + tag for tag in TARGET_TAGS.values()}
    if stack_ids != expected_ids:
        errors.append(f"config/stack.yaml targets mismatch: {sorted(stack_ids)}")

    if not args.skip_ollama:
        installed = _ollama_tags()
        missing = sorted((set(TARGET_TAGS.values()) | SUPPORT_TAGS) - installed)
        if missing:
            errors.append("missing Ollama tags: " + ", ".join(missing))

    transfer_cfg_root = REPO_ROOT / "artifacts" / "rerun_fixed_budget" / f"budget_{args.budget}" / "configs" / "transfer" / "jailbreakhub"
    for method in EXPECTED_METHODS:
        cfg_path = transfer_cfg_root / f"{method}.yaml"
        if not cfg_path.exists():
            errors.append(f"missing transfer config: {cfg_path}")
            continue
        cfg = _load_yaml(cfg_path)
        ds = cfg.get("datasets", {}).get("test_jbb_ood", {})
        if ds.get("seeds_path") != "artifacts/transfer/seeds_ood-jailbreakhub.jsonl":
            errors.append(f"{method} transfer config does not point to JailbreakHub seeds")
        if cfg.get("run_matrix", {}).get("query_budgets") != [args.budget]:
            errors.append(f"{method} transfer query_budgets is not [{args.budget}]")

    coverage = REPO_ROOT / "artifacts" / "revision_tables" / "paper_tables" / "paper_table_coverage.md"
    if not coverage.exists():
        warnings.append("paper table coverage file has not been generated yet; run scripts.build_revision_tables once")

    archived_sac = REPO_ROOT / "scripts" / "eval_old_sac_checkpoints.py"
    if not archived_sac.exists():
        errors.append("missing archived SAC eval helper: scripts/eval_old_sac_checkpoints.py")

    for script in ["scripts/rerun_fixed_budget.py", "scripts/build_revision_tables.py", "scripts/check_rerun_ready.py", "scripts/export_transfer_seeds.py", "scripts/eval_old_sac_checkpoints.py"]:
        res = subprocess.run([sys.executable, "-m", "py_compile", script], cwd=REPO_ROOT)
        if res.returncode != 0:
            errors.append(f"compile failed: {script}")

    for warning in warnings:
        print("WARN", warning)
    if errors:
        for error in errors:
            print("ERROR", error)
        return 1
    print(f"READY budget={args.budget}: eval-only SAC manifest/config/table scripts pass preflight")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
