#!/usr/bin/env python3
"""Append GPT-OSS to the live Qwen study without interrupting a partial setting."""
import json, os, time
from pathlib import Path
import run_rewriter_sensitivity as study

ROOT=study.ROOT
OUT=ROOT/'artifacts/ablations/S1_sensitivity_20261101'
RESULTS=OUT/'rewriter'
TARGET=('ollama/gpt-oss:20b','ollama_gpt-oss_20b/20250925_090542')

def main():
    # The existing Qwen worker must finish before this worker uses GPU 1.
    while len(list((RESULTS/'qwen2_5_14b').glob('seed_*/*/metrics.json')))<6:
        p=RESULTS/'status.json'
        state=json.loads(p.read_text()) if p.exists() else {}
        if state.get('state')=='failed':raise RuntimeError('Qwen evaluation failed; extension not started.')
        time.sleep(15)
    while json.loads((OUT/'status.json').read_text()).get('state')!='complete':
        time.sleep(2)
    study.TARGETS['gpt_oss_20b']=TARGET
    study.write(OUT/'status.json',{'state':'running_gptoss_extension','updated_at':time.time()})
    try:
        for seed in [1,2,3]:
            for condition in study.CONDITIONS:
                study.run_one(RESULTS,'gpt_oss_20b',condition,seed)
        study.summarize(RESULTS)
        assert len(list(RESULTS.glob('*/seed_*/*/metrics.json')))==12
        study.write(RESULTS/'status.json',{'state':'complete','completed_settings':12,'updated_at':time.time()})
        study.write(OUT/'status.json',{'state':'complete','completed_settings':12,'updated_at':time.time()})
        print('Both targets complete: 12 settings, 600 objective evaluations.',flush=True)
    except Exception as exc:
        study.write(OUT/'status.json',{'state':'failed','error':str(exc),'updated_at':time.time()});raise

if __name__=='__main__':main()
