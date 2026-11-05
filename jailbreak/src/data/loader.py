# src/data/loader.py
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

# ================================================================
# Utilities
# ================================================================


def _ensure_exists(p: Path, what: str):
    if not p.exists():
        raise FileNotFoundError(f"[data] {what} not found at: {p}")


def _json_dump(path: Path, obj: Any):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _sha256(p: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# ================================================================
# M2 SNAPSHOTS (strict on core sources, robust fallbacks)
# ================================================================

# -------- SORRY-Bench (base) seeds --------


def find_sorry_base_question(sorry_root: Path) -> Path:
    preferred = sorry_root / "base" / "question.jsonl"
    if preferred.exists():
        return preferred

    hits = list(sorry_root.rglob("question.jsonl"))
    if not hits:
        raise FileNotFoundError(
            f"[data] SORRY-Bench base/question.jsonl not found under: {sorry_root}"
        )
    return sorted(hits)[0]


def _extract_seed_text(row: Dict[str, Any]) -> str:
    for k in ("text", "question", "prompt"):
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v

    cat = row.get("category") or "Uncategorized"
    style = row.get("prompt_style") or ""
    turns = row.get("turns")
    return f"[SAFE SEED] category={cat} style={style} turns={turns}"


def snapshot_sorry_seeds(sorry_root: Path, out_path: Path) -> int:
    qfile = find_sorry_base_question(sorry_root)

    rows: List[Dict[str, Any]] = []
    with qfile.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            rows.append(
                {
                    "question_id": ex.get("question_id"),
                    "category": ex.get("category"),
                    "turns": ex.get("turns"),
                    "prompt_style": ex.get("prompt_style"),
                    "text": _extract_seed_text(ex),
                }
            )

    if not rows:
        raise RuntimeError("[data] SORRY-Bench seeds file parsed but contained 0 rows.")

    out_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8"
    )
    return len(rows)


# -------- SORRY-Bench (mutations/operators) --------


def snapshot_operator_families(sorry_root: Path, out_path: Path) -> int:
    pat = re.compile(r"question_(.+?)\.jsonl$", re.IGNORECASE)
    fams = set()

    for fp in sorry_root.rglob("question_*.jsonl"):
        m = pat.search(fp.name)
        if m:
            fams.add(m.group(1))

    if not fams:
        raise RuntimeError(
            "[data] No SORRY-Bench mutation files found (question_*.jsonl). "
            "These define operator/mutation families."
        )

    fam_list = sorted(fams)
    payload = {"operators": [{"operator_id": i, "name": n} for i, n in enumerate(fam_list)]}
    _json_dump(out_path, payload)
    return len(fam_list)


# -------- JBB categories --------


def jbb_category_counts(jbb_root: Path) -> Dict[str, Any]:
    harmful = jbb_root / "data" / "harmful-behaviors.csv"
    benign = jbb_root / "data" / "benign-behaviors.csv"

    _ensure_exists(harmful, "JBB harmful-behaviors.csv")
    _ensure_exists(benign, "JBB benign-behaviors.csv")

    df_h = pd.read_csv(harmful)
    df_b = pd.read_csv(benign)

    cats = sorted(
        set(df_h["Category"].astype(str).str.strip())
        | set(df_b["Category"].astype(str).str.strip())
    )

    return {
        "harmful_rows": int(len(df_h)),
        "benign_rows": int(len(df_b)),
        "unique_categories": int(len(cats)),
        "categories": cats,
    }


def snapshot_categories(jbb_root: Path, out_path: Path) -> Dict[str, Any]:
    harmful = jbb_root / "data" / "harmful-behaviors.csv"
    benign = jbb_root / "data" / "benign-behaviors.csv"

    _ensure_exists(harmful, "JBB harmful-behaviors.csv")
    _ensure_exists(benign, "JBB benign-behaviors.csv")

    cats: List[str] = []
    cats += pd.read_csv(harmful)["Category"].astype(str).str.strip().tolist()
    cats += pd.read_csv(benign)["Category"].astype(str).str.strip().tolist()
    cats = sorted(set(cats))

    if not cats:
        raise RuntimeError("[data] JBB category CSVs parsed but produced 0 unique categories.")

    payload = {
        "categories": [{"category_id": i, "name": name} for i, name in enumerate(cats)],
        "counts": {"unique_categories": len(cats)},
    }
    _json_dump(out_path, payload)
    return payload


# -------- MHJ pacing (multi-turn) --------


def _mhj_turns_from_messages_df(df: pd.DataFrame) -> List[int]:
    msg_cols = [c for c in df.columns if c.lower().startswith("message_")]
    if not msg_cols:
        return []

    m = df[msg_cols]
    non_empty = (~m.isna()) & (m.astype(str).str.strip() != "")
    counts = non_empty.sum(axis=1).astype(int)
    turns = counts[counts > 0].tolist()
    return [int(x) for x in turns]


def _pacing_payload_from_turns(turns: List[int], source: str) -> Dict[str, Any]:
    s = pd.Series(turns, dtype="int64")
    q25, med, q75 = map(int, s.quantile([0.25, 0.5, 0.75]).round().tolist())
    return {
        "turns_quantiles": {"q25": q25, "q50": med, "q75": q75},
        "T_default": med,
        "bounds": {"min": int(s.min()), "max": int(s.max())},
        "n_samples": int(len(turns)),
        "source": source,
    }


def snapshot_pacing(mhj_root: Path, out_path: Path, seeds_path: Path) -> Dict[str, Any]:
    _ensure_exists(mhj_root, "MHJ root directory")

    turns: List[int] = []
    source_used = None

    # Strict parse first
    for fp in list(mhj_root.rglob("*.jsonl")) + list(mhj_root.rglob("*.csv")):
        try:
            if fp.suffix == ".jsonl":
                with fp.open("r", encoding="utf-8") as f:
                    for ln in f:
                        if not ln.strip():
                            continue
                        ex = json.loads(ln)
                        v = ex.get("turns") or ex.get("num_turns")
                        if isinstance(v, int) and v > 0:
                            turns.append(int(v))
                if turns:
                    source_used = f"{fp.name}:turns/num_turns"
            else:
                df = pd.read_csv(fp)

                # 1) canonical columns
                cand: Optional[List[int]] = None
                for col in ("turns", "num_turns"):
                    if col in df.columns:
                        vals = df[col].dropna().astype(int).tolist()
                        cand = [int(x) for x in vals if int(x) > 0]
                        if cand:
                            source_used = f"{fp.name}:{col}"
                            break

                # 2) broader message_* family (user_*, assistant_*, system_*)
                if not cand:
                    msg_like = [
                        c
                        for c in df.columns
                        if re.match(r"(?i)(message|user|assistant|system)_\d+", str(c))
                    ]
                    if msg_like:
                        m = df[msg_like]
                        non_empty = (~m.isna()) & (m.astype(str).str.strip() != "")
                        cnt = non_empty.sum(axis=1).astype(int)
                        cand = [int(x) for x in cnt.tolist() if int(x) > 0]
                        if cand:
                            source_used = f"{fp.name}:message-like"

                # 3) single-message column implies 1 turn per non-empty row
                if not cand and "submission_message" in df.columns:
                    non_empty_rows = df["submission_message"].astype(str).str.strip() != ""
                    n = int(non_empty_rows.sum())
                    if n > 0:
                        cand = [1] * n
                        source_used = f"{fp.name}:submission_message->1turn"

                # 4) 'conversation' column with JSON/text list
                if not cand and "conversation" in df.columns:
                    cvals = []
                    for x in df["conversation"]:
                        try:
                            if pd.isna(x):
                                continue
                        except Exception:
                            pass

                        s = str(x).strip()
                        if not s:
                            continue

                        turns_guess = 0
                        try:
                            obj = json.loads(s)
                            if isinstance(obj, list):
                                turns_guess = len(obj)
                            elif isinstance(obj, dict) and "messages" in obj and isinstance(
                                obj["messages"], list
                            ):
                                turns_guess = len(obj["messages"])
                        except Exception:
                            turns_guess = max(1, s.count("\n") + 1)

                        if turns_guess > 0:
                            cvals.append(int(turns_guess))

                    if cvals:
                        cand = cvals
                        source_used = f"{fp.name}:conversation"

                if cand:
                    turns.extend(cand)
        except Exception:
            continue

    # If we still have nothing, fall back to seeds: 1 turn per seed
    if not turns:
        try:
            n_seeds = sum(1 for _ in Path(seeds_path).open("r", encoding="utf-8"))
            if n_seeds <= 0:
                n_seeds = 100
        except Exception:
            n_seeds = 100

        turns = [1] * n_seeds
        payload = _pacing_payload_from_turns(turns, source=f"fallback_from_seeds({n_seeds})")
        _json_dump(out_path, payload)
        return payload

    payload = _pacing_payload_from_turns(turns, source=source_used or "mixed")
    _json_dump(out_path, payload)
    return payload


# -------- Provenance --------


def snapshot_provenance(data_root: Path, run_dir: Path, files: Dict[str, Path]) -> Dict[str, Any]:
    prov = {"sources": {}, "notes": "Hashes of input files used for this run."}
    for key, p in files.items():
        prov["sources"][key] = {
            "exists": p.exists(),
            "path": str(p),
            "sha256": _sha256(p) if p.exists() and p.is_file() else None,
            "size": p.stat().st_size if p.exists() and p.is_file() else None,
        }
    _json_dump(run_dir / "dataset_provenance.json", prov)
    return prov


# -------- Optional: SORRY-Bench Human-Judgment (index only for judge training) --------


def snapshot_hj_index(sorry_hj_root: Path, out_path: Path) -> Dict[str, Any]:
    if not sorry_hj_root.exists():
        payload = {"exists": False, "root": str(sorry_hj_root), "files": []}
        _json_dump(out_path, payload)
        return payload

    files = []
    for fp in sorted(sorry_hj_root.rglob("*")):
        if fp.is_file() and fp.suffix.lower() in (".jsonl", ".csv"):
            files.append({"path": str(fp), "size": fp.stat().st_size})

    payload = {"exists": True, "root": str(sorry_hj_root), "files": files[:50]}
    _json_dump(out_path, payload)
    return payload


# ================================================================
# Orchestrator (TRAINING SNAPSHOTS)
# ================================================================


def build_training_snapshots(
    data_root: Path,
    run_dir: Path,
    *,
    include_hj_index: bool = True,
) -> Dict[str, Any]:
    sorry_root = data_root / "sorrybench"
    jbb_root = data_root / "jbb"
    mhj_root = data_root / "mhj"
    hj_root = data_root / "sorrybench_hj"

    seeds_path = run_dir / "seeds.jsonl"
    ops_path = run_dir / "operators.json"
    cats_path = run_dir / "categories.json"
    pace_path = run_dir / "pacing.json"
    hj_index = run_dir / "hj_index.json"

    run_dir.mkdir(parents=True, exist_ok=True)

    seeds_n = snapshot_sorry_seeds(sorry_root, seeds_path)
    ops_n = snapshot_operator_families(sorry_root, ops_path)
    cats = snapshot_categories(jbb_root, cats_path)
    pacing = snapshot_pacing(mhj_root, pace_path, seeds_path)

    if include_hj_index:
        snapshot_hj_index(hj_root, hj_index)

    files = {
        "sorry_base_question": (sorry_root / "base" / "question.jsonl"),
        "jbb_harmful": (jbb_root / "data" / "harmful-behaviors.csv"),
        "jbb_benign": (jbb_root / "data" / "benign-behaviors.csv"),
        "mhj_root": mhj_root,
        "hj_root": hj_root,
    }
    snapshot_provenance(data_root, run_dir, files)

    return {
        "seeds_rows": seeds_n,
        "operators": ops_n,
        "categories": cats.get("counts", {}).get("unique_categories", 0),
        "pacing": pacing,
    }


# Back-compat for existing CLI: keep the old name
def build_all(data_root: Path, run_dir: Path) -> Dict[str, Any]:
    return build_training_snapshots(data_root, run_dir, include_hj_index=True)


# ================================================================
# M5 HELPERS (unchanged interfaces)
# ================================================================


def _run_dir_from_store(rs) -> Path:
    p = getattr(rs, "run_path", None) or getattr(rs, "root", None)
    if not p:
        raise AttributeError("RunStore must expose `.run_path` or `.root`.")
    return Path(p)


def load_run_file(rs, filename: str) -> Union[List[Any], Dict[str, Any]]:
    p = _run_dir_from_store(rs) / filename
    if not p.exists():
        raise FileNotFoundError(f"Run file not found: {p}")

    if p.suffix == ".jsonl":
        rows: List[Any] = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rows.append(json.loads(line))
        return rows

    if p.suffix == ".json":
        obj: Union[Dict[str, Any], List[Any]] = json.loads(p.read_text(encoding="utf-8"))
        if filename == "operators.json":
            if isinstance(obj, dict) and "operators" in obj and isinstance(obj["operators"], list):
                return obj["operators"]
            if isinstance(obj, list):
                return obj
            raise ValueError("operators.json has unexpected shape.")
        return obj

    raise ValueError(f"Unsupported run file type: {p.name}")


def load_seeds_for_m5(rs) -> List[Dict[str, Any]]:
    seeds_raw = load_run_file(rs, "seeds.jsonl")
    out: List[Dict[str, Any]] = []

    for ex in seeds_raw:  # type: ignore[arg-type]
        ex = dict(ex)
        if not isinstance(ex.get("text"), str) or not ex["text"].strip():
            ex["text"] = _extract_seed_text(ex)
        out.append(ex)

    if not out:
        raise RuntimeError("seeds.jsonl loaded but is empty.")
    return out


def load_operators_for_m5(rs) -> List[Dict[str, Any]]:
    ops_raw = load_run_file(rs, "operators.json")
    assert isinstance(ops_raw, list), "operators.json should load to a list after unwrapping."

    norm: List[Dict[str, Any]] = []
    for i, o in enumerate(ops_raw):
        if isinstance(o, dict):
            name = o.get("name") or str(o.get("operator") or f"op_{i}")
            oid = o.get("operator_id", i)
        else:
            name, oid = str(o), i
        norm.append({"operator_id": int(oid), "name": str(name)})

    if not norm:
        raise RuntimeError("operators.json loaded but produced 0 operators.")
    return norm


def validate_m5_ready(rs) -> Tuple[int, int]:
    seeds = load_seeds_for_m5(rs)
    ops = load_operators_for_m5(rs)

    if not seeds:
        raise RuntimeError("M5: seeds.jsonl is empty. Run `run-init` to snapshot SAFE seeds first.")
    if not ops:
        raise RuntimeError(
            "M5: operators.json is empty. Check your SORRY mutations files (question_*.jsonl)."
        )
    return len(seeds), len(ops)


# Robust JSONL reader for seeds with flexible key names
def _load_seeds_jsonl(path: Path) -> List[Dict[str, Any]]:
    """
    Load a JSONL file with rows containing {question_id?, category?, text?}.
    Accepts common aliases for robustness.
    """
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            if not ln.strip():
                continue
            try:
                ex = json.loads(ln)
            except Exception:
                continue

            qid = ex.get("question_id") or ex.get("id") or ex.get("Id") or ex.get("ID")
            txt = ex.get("text") or ex.get("prompt") or ex.get("question")
            cat = ex.get("category") or "Uncategorized"

            if isinstance(txt, str) and txt.strip():
                rows.append({"question_id": qid, "category": cat, "text": txt})

    if not rows:
        raise RuntimeError(f"[data] seeds override loaded 0 usable rows from: {path}")
    return rows


# ================================================================
# OOD seed exporters
# ================================================================


def _first_nonempty(d: Dict[str, Any], keys: list[str]) -> Optional[str]:
    for k in keys:
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


JBB_GOAL_SEEDS_FILENAME = "seeds_ood-jbb-goal-v2.jsonl"


def export_jbb_harmful_to_seeds_ood(jbb_root: Path, out_path: Path) -> int:
    """
    Read JailbreakBench harmful-behaviors.csv and emit a full-goal snapshot with:
      {"question_id": "...", "category": "...", "text": "..."}
    Goal is the full evaluation request; Behavior is descriptive metadata.
    Existing differing snapshots are never silently overwritten.
    """
    src = jbb_root / "data" / "harmful-behaviors.csv"
    _ensure_exists(src, "JBB harmful-behaviors.csv")
    df = pd.read_csv(src)

    cat_cols = ["Category", "category", "Group", "group"]
    if "Goal" not in df.columns:
        raise ValueError(f"[OOD] Missing required Goal column in {src}")

    rows = []
    for i, row in df.iterrows():
        d = {str(k): row[k] for k in df.columns}
        cat = _first_nonempty(d, cat_cols) or "Uncategorized"
        goal = d["Goal"]
        if not isinstance(goal, str) or not goal.strip():
            raise ValueError(f"[OOD] Empty or invalid Goal at row {i} in {src}")
        txt = goal.strip()
        qid = str(d.get("BehaviorID") or d.get("ID") or d.get("Id") or f"jbb_{i}")
        rows.append({
            "question_id": qid,
            "category": cat,
            "text": txt,
            "behavior": _first_nonempty(d, ["Behavior", "behavior"]),
            "text_source": "Goal",
        })

    if not rows:
        raise RuntimeError("[OOD] JBB exporter found 0 usable rows.")

    content = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows)
    if out_path.exists():
        if out_path.read_text(encoding="utf-8") != content:
            raise FileExistsError(
                f"[OOD] Refusing to overwrite differing seed snapshot: {out_path}. "
                f"Export to a new versioned path such as {JBB_GOAL_SEEDS_FILENAME}."
            )
        return len(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("x", encoding="utf-8") as handle:
        handle.write(content)
    return len(rows)


