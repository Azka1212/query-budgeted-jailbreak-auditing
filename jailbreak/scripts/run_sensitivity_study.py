#!/usr/bin/env python3
"""Offline length-normalization audit, followed by queued rewriter evaluation."""
import csv, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parent))
from jailbreak.src.rl.hybrid_sac import Actor, to01
from jailbreak.src.rl.envs.prompt_hybrid_env import _hash01

def main():
    out=ROOT/'artifacts/ablations/S1_sensitivity_20261101';out.mkdir(parents=True,exist_ok=True)
    source=ROOT/'artifacts/ablations/C1_controller_feedback_20261030'
    protocol=json.loads((source/'protocol.json').read_text())
    objectives=[json.loads(l) for l in (source/'objectives.jsonl').open()]
    rows=[]
    for target,checkpoint in protocol['checkpoints'].items():
        ck=torch.load(checkpoint['path'],map_location='cpu',weights_only=False)
        actor=Actor(ck['obs_dim'],ck['num_ops'],ck['cont_dim']);actor.load_state_dict(ck['actor']);actor.eval()
        raw=json.loads((Path(checkpoint['path']).parents[2]/'operators.json').read_text());ops=raw['operators']
        states=[(i,o['name'],x) for i,x in enumerate(objectives) for o in ops]
        actions={}
        with torch.no_grad():
            for limit in [4000,8000,16000]:
                obs=np.array([[_hash01(x.get('category','Uncategorized')),_hash01(op),min(len(x['text'])/limit,1),1.] for _,op,x in states],dtype=np.float32)
                logits,mu,_=actor(torch.tensor(obs));actions[limit]=(logits.argmax(-1).numpy(),to01(mu).numpy())
        baseline=actions[8000]
        for limit,(op,cont) in actions.items():
            rows.append({'target':target,'L_max':limit,'states':len(states),
                         'operator_changes':int(np.sum(op!=baseline[0])),
                         'max_absolute_control_change':float(np.max(np.abs(cont-baseline[1]))),
                         'scope':'all 50 objectives crossed with all 20 operator hints; feedback=1',
                         'checkpoint_sha256':checkpoint['sha256']})
    with (out/'lmax_action_sensitivity.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (out/'offline_protocol.json').write_text(json.dumps({'scales':[4000,8000,16000],
        'baseline':8000,'objective_sha256':protocol['objective_sha256'],
        'checkpoints':protocol['checkpoints'],'model_calls':0,
        'scope':'evaluation-time deterministic action sensitivity, not retraining sensitivity'},indent=2))
    print('Offline L_max results:',json.dumps(rows),flush=True)
    waiting=ROOT/'artifacts/ablations/C2_one_step_controller_20261101/status.json'
    while os.getenv("SENSITIVITY_SKIP_QUEUE") != "1":
        status=json.loads(waiting.read_text())
        if status['state']=='complete':break
        if status['state']=='failed':raise RuntimeError('Existing experiment failed; inspect before scheduling more GPU calls.')
        (out/'status.json').write_text(json.dumps({'state':'queued','waiting_for':str(waiting),'updated_at':time.time()}))
        time.sleep(30)
    (out/'status.json').write_text(json.dumps({'state':'running_rewriter_sensitivity','updated_at':time.time()}))
    subprocess.run([sys.executable,'-u',str(ROOT/'scripts/run_rewriter_sensitivity.py'),
                    '--out',str(out/'rewriter')],cwd=ROOT,check=True)
    (out/'status.json').write_text(json.dumps({'state':'complete','updated_at':time.time()}))

if __name__=='__main__':main()
