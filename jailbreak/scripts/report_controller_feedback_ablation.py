#!/usr/bin/env python3
"""Create a figure only from all 24 completed evaluation settings."""
import argparse,json,time
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--wait',action='store_true');a=p.parse_args()
    while True:
        status=json.loads((a.out/'status.json').read_text()) if (a.out/'status.json').exists() else {}
        if status.get('state')=='failed':raise SystemExit('Evaluation failed; refusing to plot incomplete results.')
        if status.get('state')=='complete':break
        if not a.wait:raise SystemExit('Evaluation not complete.')
        time.sleep(30)
    rows=json.loads((a.out/'per_run_results.json').read_text());assert len(rows)==24
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    targets=[('qwen2_5_14b','Qwen2.5-14B'),('gpt_oss_20b','GPT-OSS-20B')]
    conditions=[('sac','SAC'),('random','Random'),('fixed','Fixed'),('sac_no_feedback','SAC without updated feedback')]
    fig,axes=plt.subplots(1,2,figsize=(9,3.8),sharex=True,sharey=True, squeeze=False)
    for ax,(target,title) in zip(axes.flat,targets):
        for cond,label in conditions:
            group=[r for r in rows if r['target']==target and r['condition']==cond];assert len(group)==3
            values=np.asarray([r['curve'] for r in group]);mu=values.mean(0);sd=values.std(0,ddof=1);x=np.arange(1,26)
            line,=ax.plot(x,mu,label=label);ax.fill_between(x,np.clip(mu-sd,0,1),np.clip(mu+sd,0,1),color=line.get_color(),alpha=.15)
        ax.set_title(title);ax.set_ylim(0,1.02);ax.set_xlim(1,25);ax.set_xticks([1,5,10,15,20,25]);ax.grid(alpha=.2)
    for ax in axes[-1]:ax.set_xlabel('Target queries')
    for ax in axes[:,0]:ax.set_ylabel('Cumulative ASR')
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=2,frameon=False)
    fig.tight_layout(rect=[0,.11,1,1]);fig.savefig(a.out/'controller_feedback_asr.pdf');fig.savefig(a.out/'controller_feedback_asr.png',dpi=300);plt.close(fig)
    print('Generated figure from 24 completed settings.')

if __name__=='__main__':main()
