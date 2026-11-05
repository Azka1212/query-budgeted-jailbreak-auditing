#!/usr/bin/env python3
"""Evaluate archived SAC checkpoints from the original jailbreak folder.

This avoids retraining SAC during paper revision. It runs fresh evaluation in the
clean rerun repo and writes table-ready metrics for:
  - main/guard SAC rows: artifacts/eval/<target_slug>/metrics_OOD-JBB.json
  - transfer SAC rows: artifacts/transfer/jailbreakhub/sac/models/<target>/<Q>/run_000/metrics.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
OLD_ROOT = Path("/home/security/Azka_container/jailbreak")

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

OLD_RUN_DIRS = {
    "mistral_instruct": OLD_ROOT / "artifacts/runs/ollama_mistral_instruct/20250822_045654",
    "qwen2_5_14b": OLD_ROOT / "artifacts/runs/ollama_qwen2.5_14b-instruct/20251015_075644",
    "deepseek_r1_7b": OLD_ROOT / "artifacts/runs/ollama_deepseek-r1_7b/20251015_074118",
    "vicuna_13b": OLD_ROOT / "artifacts/runs/ollama_vicuna_13b/20251007_060048",
    "gpt_oss_20b": OLD_ROOT / "artifacts/runs/ollama_gpt-oss-20b/20260109_090650",
    "llama_guard3_8b": OLD_ROOT / "artifacts/runs/ollama_llama-guard3_8b/20250908_180732",
    "granite_guardian_8b": OLD_ROOT / "artifacts/runs/ollama_granite3-guardian_8b/20250908_180531",
    "shieldgemma_9b": OLD_ROOT / "artifacts/runs/ollama_shieldgemma_9b/20250908_181001",
}

STANDARD_TARGETS = ["mistral_instruct", "deepseek_r1_7b", "qwen2_5_14b", "gpt_oss_20b", "vicuna_13b"]
ALL_TARGETS = list(TARGET_IDS)


def _slug(model_id: str) -> str:
    return re.sub(r"[^a-z0-9._-]+", "_", model_id.replace("/", "_").replace(":", "_").strip().lower()).strip("_")


def _checkpoint_for(target: str) -> Path:
    run_dir = OLD_RUN_DIRS[target]
    latest = run_dir / "rl_sac/checkpoints/ckpt_latest.pt"
    if latest.exists():
        return latest
    ckpts = list((run_dir / "rl_sac/checkpoints").glob("step_*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint for {target} under {run_dir}")
    return max(ckpts, key=lambda p: p.stat().st_mtime)


def _jbb_goal_seeds() -> Path:
    """Use the same full-goal snapshot as the corrected baseline configs."""
    from jailbreak.src.data.loader import (
        JBB_GOAL_SEEDS_FILENAME, export_jbb_harmful_to_seeds_ood,
    )
    jbb_root = REPO_ROOT / "data" / "jbb"
    path = jbb_root / "data" / JBB_GOAL_SEEDS_FILENAME
    export_jbb_harmful_to_seeds_ood(jbb_root, path)
    return path


def _run(cmd: list[str], *, env: dict[str, str]) -> None:
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _pick(d: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in d:
            return d[key]
    lower = {str(k).lower(): v for k, v in d.items()}
    for key in keys:
        if key.lower() in lower:
            return lower[key.lower()]
    return default


def _normalize_metrics(raw_path: Path, target_short: str, target_id: str, budget: int, dataset: str, out_path: Path) -> None:
    raw = _read_json(raw_path)
    counts = raw.get("counts", {}) if isinstance(raw.get("counts"), dict) else {}
    metrics = raw.get("metrics", {}) if isinstance(raw.get("metrics"), dict) else {}
    judge = raw.get("judge_means", {}) if isinstance(raw.get("judge_means"), dict) else {}
    qps = _pick(metrics, "QueriesPerSuccess", "qps", "q_per_success")
    if isinstance(qps, float) and math.isinf(qps):
        qps = "inf"
    out = {
        "method": "SAC-Triad",
        "target_short_name": target_short,
        "target_id": target_id,
        "query_budget": int(budget),
        "transfer_dataset": dataset,
        "num_pairs": _pick(counts, "pairs", "num_pairs"),
        "num_success": _pick(counts, "successes", "num_success"),
        "total_queries": _pick(counts, "total_target_calls", "total_queries"),
        "asr": _pick(metrics, "ASR", "asr"),
        "qps": qps,
        "avg_success": _pick(judge, "Success", "success"),
        "avg_stealth": _pick(judge, "Stealth", "stealth"),
        "avg_novelty": _pick(judge, "Novelty", "novelty"),
        "avg_efficiency": _pick(judge, "Efficiency", "efficiency"),
        "avg_impact": _pick(judge, "Impact", "impact"),
        "distinct_1": _pick(metrics, "Distinct1", "distinct_1"),
        "distinct_2": _pick(metrics, "Distinct2", "distinct_2"),
        "distinct_3": _pick(metrics, "Distinct3", "distinct_3"),
        "source_metrics": raw_path.as_posix(),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[old-sac] wrote normalized metrics -> {out_path}", flush=True)


def _eval_one(
    *,
    target_short: str,
    seeds: Path,
    dataset_tag: str,
    episodes: int,
    budget: int,
    run_suffix: str,
    raw_out: Path,
    normalized_out: Path | None,
    transfer_dataset: str = "",
    checkpoint_source: str | None = None,
) -> None:
    target_id = TARGET_IDS[target_short]
    ckpt = _checkpoint_for(checkpoint_source or target_short)
    if not seeds.exists():
        raise FileNotFoundError(seeds)

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT.parent}{os.pathsep}{env.get('PYTHONPATH', '')}"
    env["PYTHONUNBUFFERED"] = "1"
    env["SAC_TRIAD_RUN_ID"] = f"old_sac_{run_suffix}_q{budget}_n{episodes}_{target_short}_{int(time.time())}"

    print("\n" + "=" * 80, flush=True)
    print(f"[old-sac] target={target_short} target_id={target_id}", flush=True)
    print(f"[old-sac] checkpoint={ckpt}", flush=True)
    print(f"[old-sac] seeds={seeds}", flush=True)
    print(f"[old-sac] dataset_tag={dataset_tag} episodes={episodes} budget_label={budget}", flush=True)

    _run([sys.executable, "-m", "src.cli", "run-init", "--model", target_id, "--config-path", "config/stack.yaml"], env=env)
    _run([
        sys.executable, "-m", "src.cli", "eval-sac",
        "--checkpoint", ckpt.as_posix(),
        "--config-path", "config/stack.yaml",
        "--runs-model", target_id,
        "--target-model", target_id,
        "--use-target-echo",
        "--seeds-path-override", seeds.as_posix(),
        "--dataset-tag", dataset_tag,
        "--episodes", str(episodes),
        "--sequential-seeds",
        "--max-episode-steps", str(budget),
        "--progress-every", str(max(1, episodes // 10)),
    ], env=env)
    _run([
        sys.executable, "-m", "src.cli", "metrics-ood",
        "--dataset-tag", dataset_tag,
        "--runs-model", target_id,
        "--config-path", "config/stack.yaml",
        "--out", raw_out.as_posix(),
    ], env=env)

    if normalized_out is not None:
        _normalize_metrics(raw_out, target_short, target_id, budget, transfer_dataset, normalized_out)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--budget", type=int, default=25)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--targets", nargs="+", default=ALL_TARGETS, choices=ALL_TARGETS)
    ap.add_argument("--main", action="store_true", help="Evaluate archived SAC checkpoints on old JBB-OOD seeds.")
    ap.add_argument("--transfer", action="store_true", help="Evaluate archived SAC checkpoints on transfer seeds.")
    ap.add_argument("--model-transfer", action="store_true", help="Evaluate one frozen source checkpoint across target models on JBB-OOD.")
    ap.add_argument("--source-target", choices=ALL_TARGETS, default="qwen2_5_14b")
    ap.add_argument("--transfer-seeds", type=Path, default=Path("artifacts/transfer/seeds_ood-jailbreakhub.jsonl"))
    ap.add_argument("--transfer-dataset", default="jailbreakhub")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if not args.main and not args.transfer and not args.model_transfer:
        args.main = True
    if args.budget <= 0 or args.episodes <= 0:
        raise SystemExit("--budget and --episodes must be positive")

    for target in args.targets:
        target_id = TARGET_IDS[target]
        slug = _slug(target_id)
        if args.main:
            raw_out = REPO_ROOT / "artifacts" / "eval" / slug / "metrics_OOD-JBB.json"
            _eval_one(
                target_short=target,
                seeds=_jbb_goal_seeds(),
                dataset_tag="OOD-JBB",
                episodes=args.episodes,
                budget=args.budget,
                run_suffix="multistep_main",
                raw_out=raw_out,
                normalized_out=None,
            )
        if args.transfer:
            raw_out = REPO_ROOT / "artifacts" / "eval" / slug / f"metrics_OOD-{args.transfer_dataset}.json"
            norm_out = REPO_ROOT / "artifacts" / "transfer" / args.transfer_dataset / "sac" / "models" / target / str(args.budget) / "run_000" / "metrics.json"
            _eval_one(
                target_short=target,
                seeds=args.transfer_seeds,
                dataset_tag=f"OOD-{args.transfer_dataset}",
                episodes=args.episodes,
                budget=args.budget,
                run_suffix=f"transfer_{args.transfer_dataset}",
                raw_out=raw_out,
                normalized_out=norm_out,
                transfer_dataset=args.transfer_dataset,
            )
        if args.model_transfer:
            source = args.source_target
            raw_out = REPO_ROOT / "artifacts" / "model_transfer" / source / "raw" / target / "metrics_OOD-model-transfer.json"
            norm_out = REPO_ROOT / "artifacts" / "model_transfer" / source / "models" / target / str(args.budget) / "run_000" / "metrics.json"
            _eval_one(
                target_short=target, seeds=_jbb_goal_seeds(),
                dataset_tag=f"MODEL-TRANSFER-{source}", episodes=args.episodes, budget=args.budget,
                run_suffix=f"model_transfer_{source}", raw_out=raw_out, normalized_out=norm_out,
                transfer_dataset=f"model_transfer:{source}", checkpoint_source=source,
            )


if __name__ == "__main__":
    main()
