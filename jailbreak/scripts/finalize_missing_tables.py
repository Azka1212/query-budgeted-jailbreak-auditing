#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, math
from pathlib import Path
from statistics import mean
ROOT=Path(__file__).resolve().parents[1]; PAPER=ROOT/"artifacts/revision_tables/paper_tables"; DOC=PAPER/"doc"; BS=chr(92)
def num(x):
    try: x=float(x); return x if math.isfinite(x) else None
    except Exception: return None
def avg(rows,key):
    xs=[num(r.get(key)) for r in rows]; xs=[x for x in xs if x is not None]; return mean(xs) if xs else float("nan")
def write_csv(path,fields,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
def write_general_tex(rows,budget):
    fields=["Setting","ASR","Stealth","Q/S","D-2"]
    lines=[BS+"begin{table}[t]",BS+"centering",BS+"footnotesize",BS+f"caption{{{BS}textbf{{Generalization under dataset and model shift at fixed $Q={budget}$.}} Dataset transfer uses unseen prompts; model transfer freezes the Qwen-trained policy and evaluates unseen target families without retraining.}}",BS+"label{tab:generalization_compact}",BS+"begin{tabular}{lcccc}",BS+"toprule","Setting & ASR $"+BS+"uparrow$ & Stealth $"+BS+"uparrow$ & Q/S $"+BS+"downarrow$ & D-2 $"+BS+"uparrow$ "+BS+BS,BS+"midrule"]
    lines += [" & ".join(str(r.get(f,"")) for f in fields)+" "+BS+BS for r in rows]
    lines += [BS+"bottomrule",BS+"end{tabular}",BS+"end{table}"]
    DOC.mkdir(parents=True,exist_ok=True); (DOC/"gen.tex").write_text("\n".join(lines)+"\n")
def generalization(source,budget):
    paths=sorted((ROOT/f"artifacts/model_transfer/{source}/models").glob(f"*/{budget}/run_000/metrics.json")); metrics=[json.loads(p.read_text()) for p in paths]
    if not metrics: raise RuntimeError("No model-transfer metrics")
    fields=["Setting","ASR","Stealth","Q/S","D-2"]; path=PAPER/"generalization_compact.csv"; rows=list(csv.DictReader(path.open())) if path.exists() else []
    rows=[r for r in rows if not r.get("Setting","").startswith("Model transfer")]
    rows.append({"Setting":f"Model transfer ({source} frozen policy)","ASR":f"{avg(metrics,'asr'):.3f}","Stealth":f"{avg(metrics,'avg_stealth'):.3f}","Q/S":f"{avg(metrics,'qps'):.3f}","D-2":f"{avg(metrics,'distinct_2'):.3f}"}); write_csv(path,fields,rows)
    md="| "+" | ".join(fields)+" |\n| "+" | ".join("---" for _ in fields)+" |\n"+"".join("| "+" | ".join(str(r.get(f,"")) for f in fields)+" |\n" for r in rows); (PAPER/"generalization_compact.md").write_text(md); write_general_tex(rows,budget); return len(paths)
def coverage(budget,n):
    base=ROOT/"artifacts/runs/ollama_deepseek-r1_7b"; runs=sorted(base.glob(f"old_sac_main_q{budget}_n50_*"),key=lambda p:p.stat().st_mtime)
    if not runs: raise RuntimeError("No fresh DeepSeek run")
    events=runs[-1]/"events.jsonl"; judges=[]
    for line in events.open():
        e=json.loads(line)
        if e.get("phase")=="judge": judges.append(e)
        if len(judges)==n: break
    scored=sum(bool(e.get("response_meta",{}).get("parsed_ok")) and isinstance(e.get("scores"),dict) for e in judges); total=len(judges); cov=scored/total if total else 0.0
    row={"Setting":"deepseek_r1_7b (fresh Q=25 sample)","Total N":total,"Scored N":scored,"Coverage":f"{cov:.3f}","Refusal Rate":f"{1-cov:.3f}"}; write_csv(PAPER/"D2_judge_coverage.csv",list(row),[row])
    lines=[BS+"begin{table}[t]",BS+"centering",BS+"small",BS+f"caption{{{BS}textbf{{Judge scoring coverage on a fresh diagnostic subset.}} We inspect {total} DeepSeek-R1-7B outputs from the fixed-cap $Q={budget}$ evaluation.}}",BS+"label{tab:d2_judge_coverage}",BS+"begin{tabular}{lcccc}",BS+"toprule","Setting & Total N & Scored N & Coverage $"+BS+"uparrow$ & Refusal Rate $"+BS+"downarrow$ "+BS+BS,BS+"midrule",f"{row['Setting']} & {total} & {scored} & {cov:.3f} & {1-cov:.3f} "+BS+BS,BS+"bottomrule",BS+"end{tabular}",BS+"end{table}"]
    DOC.mkdir(parents=True,exist_ok=True); (DOC/"D2_judge_coverage.tex").write_text("\n".join(lines)+"\n"); return events
def main():
    p=argparse.ArgumentParser(); p.add_argument("--budget",type=int,default=25); p.add_argument("--source",default="qwen2_5_14b"); p.add_argument("--sample-size",type=int,default=25); a=p.parse_args(); print("model-transfer metrics:",generalization(a.source,a.budget)); print("judge source:",coverage(a.budget,a.sample_size))
if __name__=="__main__": main()
