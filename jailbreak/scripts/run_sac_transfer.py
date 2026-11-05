#!/usr/bin/env python3
"""Evaluate latest trained SAC checkpoints on a transfer seed set."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET_IDS = {
    "qwen2_5_14b": "ollama/qwen2.5:14b-instruct",
    "deepseek_r1_7b": "ollama/deepseek-r1:7b",
    "vicuna_13b": "ollama/vicuna:13b",
    "gpt_oss_20b": "ollama/gpt-oss:20b",
    "llama_guard3_8b": "ollama/llama-guard3:8b",
    "granite_guardian_8b": "ollama/granite3-guardian:8b",
    "shieldgemma_9b": "ollama/shieldgemma:9b",
}


def _slug(model_id: str) -> str:
    import re
    return re.sub(r"[^a-z0-9._-]+", "_", model_id.replace("/", "_").replace(":", "_").strip().lower()).strip("_")


def _load_app_config(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data.get("app", data)


def _latest_run_dir(runs_root: Path, model_id: str) -> Path:
    model_root = runs_root / _slug(model_id)
    if not model_root.exists():
        raise FileNotFoundError(f"No run directory for {model_id}: {model_root}")
    subdirs = [p for p in model_root.iterdir() if p.is_dir()]
    if not subdirs:
        raise FileNotFoundError(f"No timestamped run dirs under {model_root}")
    return max(subdirs, key=lambda p: p.stat().st_mtime)


def _latest_checkpoint(run_dir: Path) -> Path:
    ck_dir = run_dir / "rl_sac" / "checkpoints"
    ckpts = list(ck_dir.glob("step_*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No SAC checkpoints in {ck_dir}")
    return max(ckpts, key=lambda p: p.stat().st_mtime)


def _normalize_metrics(raw_path: Path, target_short: str, target_id: str, budget: int, dataset: str, out_path: Path) -> None:
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    counts = raw.get("counts", {}) or {}
    metrics = raw.get("metrics", {}) or {}
    judge = raw.get("judge_means", {}) or {}
    qps = metrics.get("QueriesPerSuccess")
    out = {
        "method": "SAC-Triad",
        "target_short_name": target_short,
        "target_id": target_id,
        "query_budget": int(budget),
        "transfer_dataset": dataset,
        "num_pairs": counts.get("pairs"),
        "num_success": counts.get("successes"),
        "total_queries": counts.get("total_target_calls"),
        "asr": metrics.get("ASR"),
        "qps": qps if qps != math.inf else "inf",
        "avg_success": judge.get("Success") or judge.get("success"),
        "avg_stealth": judge.get("Stealth") or judge.get("stealth"),
        "avg_novelty": judge.get("Novelty") or judge.get("novelty"),
        "avg_efficiency": judge.get("Efficiency") or judge.get("efficiency"),
        "avg_impact": judge.get("Impact") or judge.get("impact"),
        "distinct_1": metrics.get("Distinct1"),
        "distinct_2": metrics.get("Distinct2"),
        "distinct_3": metrics.get("Distinct3"),
        "source_metrics": raw_path.as_posix(),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config-path", type=Path, default=Path("config/stack.yaml"))
    ap.add_argument("--seeds", type=Path, required=True)
    ap.add_argument("--dataset", default="jailbreakhub")
    ap.add_argument("--budget", type=int, default=25)
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--targets", nargs="+", required=True)
    ap.add_argument("--out-root", type=Path, default=Path("artifacts/transfer/jailbreakhub/sac"))
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    app_cfg = _load_app_config(args.config_path)
    runs_root = Path(app_cfg["paths"]["runs"])
    dataset_tag = f"OOD-{args.dataset}"
    for target_short in args.targets:
        target_id = TARGET_IDS[target_short]
        run_dir = _latest_run_dir(runs_root, target_id)
        ckpt = _latest_checkpoint(run_dir)
        print(f"[sac-transfer] target={target_short} run={run_dir.name} ckpt={ckpt.name}")
        cmd = [
            sys.executable, "-m", "src.cli", "eval-sac",
            "--checkpoint", ckpt.as_posix(),
            "--config-path", args.config_path.as_posix(),
            "--runs-model", target_id,
            "--target-model", target_id,
            "--use-target-echo",
            "--seeds-path-override", args.seeds.as_posix(),
            "--dataset-tag", dataset_tag,
            "--episodes", str(args.episodes),
            "--progress-every", "20",
        ]
        subprocess.run(cmd, cwd=REPO_ROOT, check=True)
        raw_metrics = REPO_ROOT / "artifacts" / "eval" / _slug(target_id) / f"metrics_{dataset_tag}.json"
        subprocess.run([
            sys.executable, "-m", "src.cli", "metrics-ood",
            "--dataset-tag", dataset_tag,
            "--runs-model", target_id,
            "--config-path", args.config_path.as_posix(),
            "--out", raw_metrics.as_posix(),
        ], cwd=REPO_ROOT, check=True)
        out_path = args.out_root / "models" / target_short / str(args.budget) / "run_000" / "metrics.json"
        _normalize_metrics(raw_metrics, target_short, target_id, args.budget, args.dataset, out_path)
        print(f"[sac-transfer] wrote {out_path}")


if __name__ == "__main__":
    main()
