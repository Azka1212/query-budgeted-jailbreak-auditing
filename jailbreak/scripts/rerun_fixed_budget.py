#!/usr/bin/env python3
"""Build and optionally execute a fixed-query-budget paper rerun plan.

This clean-copy runner is intentionally conservative:
  * baselines receive the same hard --query-budgets value;
  * ablation YAML copies are patched to the same budget;
  * main SAC is retrained and then evaluated for every target in config/stack.yaml;
  * legacy checkpoint-only SAC evaluation is available as sac_eval, but is not
    part of the default final rerun because it depends on old checkpoint paths;
  * revision-ready CSV/Markdown tables are generated at the end.

Default mode is dry-run. Pass --execute only after the manifest looks right and
router-health passes.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyYAML is required. Install project requirements first.") from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent
RERUN_ROOT = REPO_ROOT / "artifacts" / "rerun_fixed_budget"

TARGET_IDS = {
    "qwen2_5_14b": "ollama/qwen2.5:14b-instruct",
    "deepseek_r1_7b": "ollama/deepseek-r1:7b",
    "vicuna_13b": "ollama/vicuna:13b",
    "gpt_oss_20b": "ollama/gpt-oss:20b",
    "llama_guard3_8b": "ollama/llama-guard3:8b",
    "granite_guardian_8b": "ollama/granite3-guardian:8b",
    "shieldgemma_9b": "ollama/shieldgemma:9b",
}
DEFAULT_TARGETS = list(TARGET_IDS)
STANDARD_TARGETS = ["qwen2_5_14b", "deepseek_r1_7b", "vicuna_13b", "gpt_oss_20b"]
TRANSFER_DATASETS = {"jailbreakhub": "artifacts/transfer/seeds_ood-jailbreakhub.jsonl"}

BASELINE_RUNNERS = {
    "gcg": ("src.baselines.runner", "config/baselines/gcg.yaml"),
    "autodan": ("src.baselines.autodan.runner", "config/baselines/autodan.yaml"),
    "gptfuzzer": ("src.baselines.GPTFUZZER.runner", "config/baselines/gptfuzzer.yaml"),
    "tap": ("src.baselines.TAP.runner", "config/baselines/tap.yaml"),
    "pair": ("src.baselines.PAIR.runner", "config/baselines/pair.yaml"),
    "rlbreaker": ("src.baselines.rlbreaker.runner", "config/baselines/rlbreaker.yaml"),
    "xjailbreak": ("src.baselines.xjailbreak.runner", "config/baselines/xjailbreak.yaml"),
}

B2_CONFIGS = {
    "binary": "config/ablations/b2/reward_binary.yaml",
    "scalar": "config/ablations/b2/reward_scalar.yaml",
    "full5d": "config/ablations/b2/reward_full5d.yaml",
}

SAC_EVAL_CONFIGS = {
    "qwen2_5_14b": "src/ablations/sac_triad/qwen2_5_14b.yaml",
    "deepseek_r1_7b": "src/ablations/sac_triad/deepseek_r1_7b.yaml",
    "vicuna_13b": "src/ablations/sac_triad/vicuna_13b.yaml",
    "gpt_oss_20b": "src/ablations/sac_triad/gpt_oss_20b.yaml",
    "llama_guard3_8b": "src/ablations/sac_triad/llama_guard3_8b.yaml",
    "granite_guardian_8b": "src/ablations/sac_triad/granite_guardian_8b.yaml",
    "shieldgemma_9b": "src/ablations/sac_triad/shieldgemma_9b.yaml",
}


@dataclass(frozen=True)
class PlanItem:
    group: str
    name: str
    command: list[str]
    cwd: str
    notes: str = ""


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping YAML at {path}")
    return data


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def _patch_nested_budgets(data: Any, budget: int) -> Any:
    if isinstance(data, dict):
        patched: dict[str, Any] = {}
        for key, value in data.items():
            if key in {"query_budget", "eval_budget", "budget_k", "max_query_budget"}:
                patched[key] = budget
            elif key == "query_budgets" and isinstance(value, list):
                patched[key] = [budget]
            elif key in {"model_id", "target_model", "judge_model", "rewriter_model"} and isinstance(value, str):
                patched[key] = {
                    "ollama/gpt-oss-20b": "ollama/gpt-oss:20b",
                    "ollama/llama3:judge": "ollama/llama3:instruct",
                    "ollama/yi-9b": "ollama/yi:9b",
                }.get(value, value)
            else:
                patched[key] = _patch_nested_budgets(value, budget)
        return patched
    if isinstance(data, list):
        return [_patch_nested_budgets(item, budget) for item in data]
    return data




def _patch_nested_max_pairs(data: Any, max_pairs: int | None) -> Any:
    if max_pairs is None:
        return data
    if isinstance(data, dict):
        patched: dict[str, Any] = {}
        for key, value in data.items():
            if key == "max_pairs":
                patched[key] = int(max_pairs)
            else:
                patched[key] = _patch_nested_max_pairs(value, max_pairs)
        return patched
    if isinstance(data, list):
        return [_patch_nested_max_pairs(item, max_pairs) for item in data]
    return data


def _patched_baseline_config(
    src: Path,
    *,
    method: str,
    budget: int,
    targets: list[str],
    max_pairs: int | None,
) -> Path:
    dst = RERUN_ROOT / f"budget_{budget}" / "configs" / "baselines" / f"{method}.yaml"
    data = _patch_nested_budgets(_load_yaml(src), budget)
    data = _patch_nested_max_pairs(data, max_pairs)
    data.setdefault("run_matrix", {})["query_budgets"] = [budget]
    data["run_matrix"]["target_short_names"] = list(targets)
    _write_yaml(dst, data)
    return dst

def _patched_config(src: Path, budget: int, family: str) -> Path:
    rel = src.relative_to(REPO_ROOT)
    dst = RERUN_ROOT / f"budget_{budget}" / "configs" / family / rel.name
    data = _patch_nested_budgets(_load_yaml(src), budget)
    _write_yaml(dst, data)
    return dst


def _patched_transfer_baseline_config(
    src: Path,
    *,
    method: str,
    dataset: str,
    seeds_path: str,
    budget: int,
    max_pairs: int | None,
) -> Path:
    dst = RERUN_ROOT / f"budget_{budget}" / "configs" / "transfer" / dataset / f"{method}.yaml"
    data = _patch_nested_budgets(_load_yaml(src), budget)
    data = _patch_nested_max_pairs(data, max_pairs)
    data.setdefault("paths", {})["artifacts_root"] = f"artifacts/transfer/{dataset}/baselines/{method}"
    data.setdefault("datasets", {}).setdefault("test_jbb_ood", {})["name"] = dataset
    data["datasets"]["test_jbb_ood"]["seeds_path"] = seeds_path
    data.setdefault("run_matrix", {})["dataset"] = "test_jbb_ood"
    data["run_matrix"]["experiment_name"] = f"transfer_{dataset}_{method}"
    data["run_matrix"]["query_budgets"] = [budget]
    data["run_matrix"]["target_short_names"] = list(STANDARD_TARGETS)
    _write_yaml(dst, data)
    return dst


def _python_module(module: str, *args: str) -> list[str]:
    return [sys.executable, "-m", module, *args]


def _enabled_stack_targets() -> set[str]:
    raw = _load_yaml(REPO_ROOT / "config/stack.yaml")
    app = raw.get("app", raw)
    out: set[str] = set()
    for item in app.get("targets", []):
        if isinstance(item, dict) and item.get("enabled", True):
            out.add(str(item.get("id")))
        elif isinstance(item, str):
            out.add(item)
    return out


def _baseline_short_names(config_path: Path) -> set[str]:
    raw = _load_yaml(config_path)
    names: set[str] = set()
    for item in raw.get("targets", []):
        if isinstance(item, dict) and item.get("enabled", True):
            names.add(str(item.get("short_name")))
    return names


def validate_inputs(targets: list[str], methods: list[str], groups: set[str]) -> list[str]:
    warnings: list[str] = []
    unknown = [t for t in targets if t not in TARGET_IDS]
    if unknown:
        raise SystemExit(f"Unknown target short names: {unknown}. Known: {sorted(TARGET_IDS)}")

    if "baselines" in groups:
        for method in methods:
            _, cfg = BASELINE_RUNNERS[method]
            available = _baseline_short_names(REPO_ROOT / cfg)
            missing = [t for t in targets if t not in available]
            if missing:
                raise SystemExit(f"{method} config is missing targets: {missing}")

    if "main_sac" in groups:
        enabled = _enabled_stack_targets()
        missing_ids = [TARGET_IDS[t] for t in targets if TARGET_IDS[t] not in enabled]
        if missing_ids:
            raise SystemExit(
                "config/stack.yaml is missing enabled paper targets: " + ", ".join(missing_ids)
            )

    if "sac_eval" in groups:
        warnings.append(
            "sac_eval is checkpoint-only legacy evaluation; use main_sac for fresh reproduction."
        )

    return warnings


def build_plan(
    *,
    budget: int,
    repeats: int,
    targets: list[str],
    groups: set[str],
    methods: list[str],
    sac_train_steps: int,
    sac_eval_episodes: int,
    max_pairs: int | None,
    skip_sac_transfer: bool,
) -> list[PlanItem]:
    plan: list[PlanItem] = []

    if "baselines" in groups:
        for method in methods:
            module, config_path = BASELINE_RUNNERS[method]
            cfg_path = config_path
            if max_pairs is not None:
                patched = _patched_baseline_config(
                    REPO_ROOT / config_path,
                    method=method,
                    budget=budget,
                    targets=targets,
                    max_pairs=max_pairs,
                )
                cfg_path = str(patched.relative_to(REPO_ROOT))
            cmd = _python_module(
                module,
                "--config-path", cfg_path,
                "--query-budgets", str(budget),
                "--repeats", str(repeats),
                "--targets", *targets,
            )
            note = "hard fixed query budget"
            if max_pairs is not None:
                note += f"; fixed evaluation subset max_pairs={max_pairs}"
            plan.append(PlanItem("baselines", method, cmd, str(REPO_ROOT), note))

    if "main_sac" in groups:
        plan.append(
            PlanItem(
                "main_sac",
                "train_all_targets",
                _python_module(
                    "src.cli", "train-all",
                    "--config-path", "config/stack.yaml",
                    "--total-steps", str(sac_train_steps),
                    "--use-target-echo",
                ),
                str(REPO_ROOT),
                "fresh SAC training for every enabled target in config/stack.yaml",
            )
        )
        plan.append(
            PlanItem(
                "main_sac",
                "test_all_targets",
                _python_module(
                    "src.cli", "test-all",
                    "--config-path", "config/stack.yaml",
                    "--episodes", str(sac_eval_episodes),
                ),
                str(REPO_ROOT),
                "fresh SAC OOD evaluation after training; writes metrics per target",
            )
        )

    if "sac_eval" in groups:
        for target in targets:
            config = SAC_EVAL_CONFIGS.get(target)
            if not config:
                continue
            patched = _patched_config(REPO_ROOT / config, budget, "sac_triad")
            cmd = _python_module(
                "src.ablations.sac_triad.runner",
                "--config-path", str(patched.relative_to(REPO_ROOT)),
            )
            plan.append(
                PlanItem(
                    "sac_eval",
                    f"checkpoint_eval:{target}",
                    cmd,
                    str(REPO_ROOT),
                    "legacy checkpoint-only eval; paths must exist before execution",
                )
            )

    if "transfer" in groups:
        dataset = "jailbreakhub"
        seeds_path = TRANSFER_DATASETS[dataset]
        plan.append(
            PlanItem(
                "transfer",
                f"export_seeds:{dataset}",
                _python_module(
                    "scripts.export_transfer_seeds",
                    "--dataset", dataset,
                    "--data-root", "data",
                    "--out", seeds_path,
                    "--max-rows", str(max_pairs or 200),
                ),
                str(REPO_ROOT),
                "export transfer dataset into common seeds JSONL format",
            )
        )
        for method in methods:
            module, config_path = BASELINE_RUNNERS[method]
            patched = _patched_transfer_baseline_config(
                REPO_ROOT / config_path,
                method=method,
                dataset=dataset,
                seeds_path=seeds_path,
                budget=budget,
                max_pairs=max_pairs,
            )
            plan.append(
                PlanItem(
                    "transfer",
                    f"{dataset}:{method}",
                    _python_module(
                        module,
                        "--config-path", str(patched.relative_to(REPO_ROOT)),
                        "--query-budgets", str(budget),
                        "--repeats", str(repeats),
                        "--targets", *STANDARD_TARGETS,
                    ),
                    str(REPO_ROOT),
                    "baseline dataset-transfer rerun with fixed query budget",
                )
            )
        if not skip_sac_transfer:
            plan.append(
                PlanItem(
                    "transfer",
                    f"{dataset}:SAC-Triad",
                    _python_module(
                        "scripts.run_sac_transfer",
                        "--config-path", "config/stack.yaml",
                        "--seeds", seeds_path,
                        "--dataset", dataset,
                        "--budget", str(budget),
                        "--episodes", str(max_pairs or 200),
                        "--targets", *STANDARD_TARGETS,
                        "--out-root", f"artifacts/transfer/{dataset}/sac",
                    ),
                    str(REPO_ROOT),
                    "evaluate latest trained SAC checkpoints on transfer dataset",
                )
            )

    if "ablations" in groups:
        b1 = _patched_config(REPO_ROOT / "config/ablations/b1_algorithm.yaml", budget, "ablations")
        b3 = _patched_config(REPO_ROOT / "config/ablations/b3_component.yaml", budget, "ablations")
        plan.append(
            PlanItem(
                "ablations",
                "b1_algorithm",
                _python_module("src.ablations.runner", "--config-path", str(b1.relative_to(REPO_ROOT))),
                str(REPO_ROOT),
                "SAC/PPO/Bandit ablation with budget-patched config",
            )
        )
        plan.append(
            PlanItem(
                "ablations",
                "b3_component",
                _python_module("src.ablations.B3.b3_runner", "--config-path", str(b3.relative_to(REPO_ROOT))),
                str(REPO_ROOT),
                "component ablation with budget-patched config",
            )
        )
        for variant, config_path in B2_CONFIGS.items():
            patched = _patched_config(REPO_ROOT / config_path, budget, "ablations_b2")
            plan.append(
                PlanItem(
                    "ablations",
                    f"b2_reward:{variant}",
                    _python_module(
                        "src.ablations.baseline.b2.b2_runner",
                        "--config-path", str(patched.relative_to(REPO_ROOT)),
                        "--stack-config-path", "config/stack.yaml",
                    ),
                    str(REPO_ROOT),
                    "reward ablation with budget-patched config",
                )
            )

    if "tables" in groups:
        plan.append(
            PlanItem(
                "tables",
                "build_revision_tables",
                _python_module(
                    "scripts.build_revision_tables",
                    "--budget", str(budget),
                    "--targets", *targets,
                    "--methods", *methods,
                ),
                str(REPO_ROOT),
                "collect completed JSON/CSV metrics into paper-ready CSV and Markdown tables",
            )
        )

    return plan


def write_manifest(plan: list[PlanItem], args: argparse.Namespace, warnings: list[str]) -> Path:
    out_dir = RERUN_ROOT / f"budget_{args.budget}"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at_unix": int(time.time()),
        "repo_root": str(REPO_ROOT),
        "budget": args.budget,
        "budget_policy": "Baselines, ablation configs, and transfer configs are hard-patched to this budget. If max_pairs is set, baseline/transfer configs and transfer seed export use the same fixed subset size. main_sac is fresh train/test over the same target set; current SAC env is one-step per episode unless multi-turn env is separately enabled.",
        "repeats": args.repeats,
        "targets": args.targets,
        "target_ids": {t: TARGET_IDS[t] for t in args.targets},
        "groups": args.groups,
        "methods": args.methods,
        "sac_train_steps": args.sac_train_steps,
        "sac_eval_episodes": args.sac_eval_episodes,
        "max_pairs": args.max_pairs,
        "skip_sac_transfer": args.skip_sac_transfer,
        "execute": args.execute,
        "warnings": warnings,
        "items": [asdict(item) for item in plan],
    }
    path = out_dir / "rerun_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def run_plan(plan: list[PlanItem], *, execute: bool) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{WORKSPACE_ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}"

    for index, item in enumerate(plan, start=1):
        printable = " ".join(item.command)
        print(f"[{index:03d}/{len(plan):03d}] {item.group}/{item.name}")
        print(f"  cwd: {item.cwd}")
        print(f"  cmd: {printable}")
        if item.notes:
            print(f"  note: {item.notes}")
        if execute:
            subprocess.run(item.command, cwd=item.cwd, env=env, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, default=25, help="Fixed query budget for baselines/ablations.")
    parser.add_argument("--repeats", type=int, default=1, help="Repeats per target/method/budget.")
    parser.add_argument(
        "--groups",
        nargs="+",
        default=["baselines", "main_sac", "ablations", "transfer", "tables"],
        choices=["baselines", "main_sac", "sac_eval", "ablations", "transfer", "tables"],
        help="Experiment groups to include.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(BASELINE_RUNNERS),
        choices=list(BASELINE_RUNNERS),
        help="Baseline methods to include when group=baselines.",
    )
    parser.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS, help="Target short names.")
    parser.add_argument("--sac-train-steps", type=int, default=20000)
    parser.add_argument("--sac-eval-episodes", type=int, default=200)
    parser.add_argument("--max-pairs", type=int, default=None, help="Optional fixed evaluation subset size for baseline and transfer configs.")
    parser.add_argument("--skip-sac-transfer", action="store_true", help="Do not add SAC transfer eval to transfer group; use archived-checkpoint eval helper instead.")
    parser.add_argument("--execute", action="store_true", help="Run commands. Omit for dry-run.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.budget <= 0:
        raise SystemExit("--budget must be positive")
    if args.repeats <= 0:
        raise SystemExit("--repeats must be positive")
    if args.sac_train_steps <= 0 or args.sac_eval_episodes <= 0:
        raise SystemExit("SAC train steps and eval episodes must be positive")
    if args.max_pairs is not None and args.max_pairs <= 0:
        raise SystemExit("--max-pairs must be positive when provided")

    groups = set(args.groups)
    warnings = validate_inputs(args.targets, args.methods, groups)
    plan = build_plan(
        budget=args.budget,
        repeats=args.repeats,
        targets=args.targets,
        groups=groups,
        methods=args.methods,
        sac_train_steps=args.sac_train_steps,
        sac_eval_episodes=args.sac_eval_episodes,
        max_pairs=args.max_pairs,
        skip_sac_transfer=args.skip_sac_transfer,
    )
    manifest = write_manifest(plan, args, warnings)
    print(f"[manifest] {manifest}")
    for warning in warnings:
        print(f"[warning] {warning}")
    run_plan(plan, execute=args.execute)
    if not args.execute:
        print("\nDry-run only. Re-run with --execute after router-health passes.")


if __name__ == "__main__":
    main()
