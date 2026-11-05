#!/usr/bin/env python3
"""Read-only empirical analysis; write separate audit artifacts, never alter experiments."""
import collections,csv,hashlib,json,math,re,sys
from datetime import datetime
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT.parents[1]/'submission_revision/reviewer1_analysis'
ART=ROOT/'artifacts/ablations'

def read(path):
    rows=[]
    if not path.exists():return rows
    for line in path.open():
        try:rows.append(json.loads(line))
        except json.JSONDecodeError:continue
    return rows

def csvout(name,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def normalized(x):return re.sub(r'\s+',' ',str(x)).strip().casefold()

def bootstrap(study,target,condition,other):
    # Cluster on objective ID: retain all three seeds together and pair conditions.
    mapping={}
    for cond in [condition,other]:
        lists=[read(study/target/f'seed_{s:03d}'/cond/'episodes.jsonl') for s in [1,2,3]]
        if any(len(x)!=50 for x in lists):return None
        mapping[cond]=[{x['objective_id']:x for x in rows} for rows in lists]
    ids=sorted(mapping[condition][0]);assert all(set(d)==set(ids) for ds in mapping.values() for d in ds)
    arrays={}
    for cond,ds in mapping.items():
        arrays[cond]=(np.array([[d[k]['first_success'] is not None for k in ids] for d in ds],float),
                      np.array([[d[k]['queries'] for k in ids] for d in ds],float))
    rng=np.random.default_rng(20261101);ix=rng.integers(0,50,(5000,50))
    def calc(cond):
        success,calls=arrays[cond];s=success[:,ix].sum(-1);q=calls[:,ix].sum(-1)
        asr=s.mean(0)/50
        qp=np.divide(q,s,out=np.full_like(q,np.nan),where=s>0).mean(0)
        orig_q=np.divide(calls.sum(1),success.sum(1),out=np.full(3,np.nan),where=success.sum(1)>0).mean()
        return asr,qp,float(success.mean()),float(orig_q)
    a=calc(condition);b=calc(other)
    row={'study':study.name,'target':target,'contrast':condition+' minus '+other,
         'objective_clusters':50,'evaluation_seeds':3,'bootstrap_replicates':5000,
         'ASR_difference':a[2]-b[2]}
    delta=a[0]-b[0];row.update(ASR_CI_low=float(np.quantile(delta,.025)),ASR_CI_high=float(np.quantile(delta,.975)))
    delta=a[1]-b[1];valid=delta[np.isfinite(delta)]
    row.update(QS_difference=a[3]-b[3],QS_valid_bootstraps=len(valid),
               QS_CI_low=float(np.quantile(valid,.025)) if len(valid)==5000 else None,
               QS_CI_high=float(np.quantile(valid,.975)) if len(valid)==5000 else None)
    return row

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    studies=[ART/'C1_controller_feedback_20261030',ART/'C2_one_step_controller_20261101',
             ART/'S1_sensitivity_20261101/rewriter']
    costs=[];policies=[];examples=[];coverage=[];statistics=[]
    for study in studies:
        for folder in sorted(study.glob('*/seed_*/*')):
            if not folder.is_dir():continue
            attempts=read(folder/'attempts.jsonl');episodes=read(folder/'episodes.jsonl');calls=read(folder/'model_calls.jsonl')
            if not attempts:continue
            target,seed,condition=folder.relative_to(study).parts
            complete=(folder/'metrics.json').exists()
            prefix={'study':study.name,'target':target,'seed':seed,'condition':condition,'complete':complete}
            cost=dict(prefix,completed_objectives=len(episodes),logged_attempts=len(attempts),
                      complete_model_call_records=sum('result' in x for x in calls),
                      failed_model_call_records=sum('error' in x for x in calls))
            for stage in ['rewriter','target','judge']:
                cs=[x for x in calls if x['stage']==stage and 'result' in x]
                cost[stage+'_calls']=len(cs)
                cost[stage+'_latency_seconds']=sum(x['result'].get('latency_ms',0) for x in cs)/1000
                cost[stage+'_input_tokens']=sum(x['result']['raw'].get('prompt_eval_count',0) for x in cs)
                cost[stage+'_output_tokens']=sum(x['result']['raw'].get('eval_count',0) for x in cs)
                cost[stage+'_empty_final']=sum(not x['result'].get('text','').strip() for x in cs)
                cost[stage+'_at_output_limit']=sum(x['result']['raw'].get('eval_count',0)>=x['request']['options']['num_predict'] for x in cs)
            cost['judge_unparsed_attempts']=sum(not x['info'].get('judge_parsed_ok',False) for x in attempts)
            costs.append(cost)
            coverage.append(dict(prefix,completed_objectives=len(episodes),expected_objectives=50))
            for op,n in collections.Counter(x['info']['operator'] for x in attempts).items():
                policies.append(dict(prefix,operator=op,attempts=n,fraction=n/len(attempts)))
            # Deterministic first successful and unsuccessful examples; no harmful text reproduced.
            for outcome in ['success','failure']:
                selected=next((e for e in episodes if (e['first_success'] is not None)==(outcome=='success')),None)
                if selected is None:continue
                trace=[a for a in attempts if a['objective_id']==selected['objective_id']]
                examples.append(dict(prefix,outcome=outcome,objective_id=selected['objective_id'],
                    category=trace[0]['info'].get('category'),queries=selected['queries'],
                    first_success=selected['first_success'],
                    operator_sequence=json.dumps([a['info']['operator'] for a in trace]),
                    success_scores=json.dumps([a['info']['scores'].get('Success',0) for a in trace]),
                    first_controls=json.dumps(trace[0]['action']['cont']),source=str(folder/'attempts.jsonl')))
        if study.name.startswith('C'):
            contrasts=['random','fixed']+(['sac_no_feedback'] if study.name.startswith('C1') else [])
            for target in ['qwen2_5_14b','gpt_oss_20b']:
                for comparator in contrasts:
                    row=bootstrap(study,target,'sac',comparator)
                    if row:statistics.append(row)
    csvout('evaluation_costs.csv',costs);csvout('policy_operator_frequencies.csv',policies)
    csvout('representative_traces.csv',examples);csvout('analysis_coverage.csv',coverage)
    csvout('paired_objective_bootstrap.csv',statistics)
    # Datasets: use checkpoint training snapshots and the frozen test objectives.
    protocol=json.loads((studies[0]/'protocol.json').read_text())
    train=read(Path(protocol['checkpoints']['qwen2_5_14b']['path']).parents[2]/'seeds.jsonl')
    test=read(studies[0]/'objectives.jsonl')
    with (ROOT/'data/jailbreakhub/data/forbidden_question/forbidden_question_set.csv').open() as f:
        hub=[{'text':x.get('question',''),'category':x.get('content_policy_name','')} for x in csv.DictReader(f)]
    sets={'SORRY_training_snapshot':train,'JBB_full_objectives_evaluated':test,'JailbreakHub_source_collection':hub}
    datasets=[];categories=[];overlaps=[]
    for name,rows in sets.items():
        texts=[x['text'] for x in rows if x.get('text','').strip()];lens=np.array([len(x) for x in texts])
        datasets.append({'dataset':name,'rows':len(rows),'nonempty':len(texts),'unique_normalized':len(set(map(normalized,texts))),
             'characters_min':int(lens.min()),'characters_median':float(np.median(lens)),
             'characters_mean':float(lens.mean()),'characters_max':int(lens.max()),
             'categories':len(set(str(x.get('category')) for x in rows))})
        for cat,n in collections.Counter(str(x.get('category')) for x in rows).items():categories.append({'dataset':name,'category':cat,'count':n})
    names=list(sets)
    for i,left in enumerate(names):
        for right in names[i+1:]:
            a=set(normalized(x['text']) for x in sets[left] if x.get('text'));b=set(normalized(x['text']) for x in sets[right] if x.get('text'))
            overlaps.append({'left':left,'right':right,'normalized_exact_overlap':len(a&b),
                             'limitation':'Exact text matching only; paraphrase/semantic overlap not established'})
    csvout('dataset_summary.csv',datasets);csvout('dataset_categories.csv',categories);csvout('dataset_exact_overlap.csv',overlaps)
    training=[]
    for target,ck in protocol['checkpoints'].items():
        run=Path(ck['path']).parents[2];p=run/'rl_sac/metrics.csv'
        if not p.exists():continue
        with p.open() as f:rows=list(csv.DictReader(f))
        relevant=[x for x in rows if x.get('step','').isdigit() and int(x['step'])<20000]
        steps=[int(x['step']) for x in relevant];counts=collections.Counter(steps)
        times=[]
        for x in relevant:
            try:times.append(datetime.fromisoformat(x['time_iso'].replace('Z','+00:00')).timestamp())
            except (ValueError,KeyError):pass
        training.append({'target':target,'checkpoint_step':ck['step'],'csv_rows_under_checkpoint_step':len(relevant),
               'unique_step_indices':len(counts),'duplicate_step_indices':sum(n>1 for n in counts.values()),
               'logged_target_calls':sum(int(float(x.get('target_calls') or 0)) for x in relevant),
               'timestamp_span_hours':(max(times)-min(times))/3600 if times else None,
               'runtime_status':'Timestamp span is not verified active GPU training time; append/restart history may be present',
               'auxiliary_cost_status':'Training prompt/judge retries and token totals not established by metrics CSV',
               'source':str(p)})
    csvout('training_cost_evidence.csv',training)
    report=['# Reviewer 1 offline analysis','',
       'No model calls, training, raw-log edits, or manuscript edits were performed. Results below describe the saved checkpoints and measured evaluations, not an independent validation of historical main-table cells.','',
       '## Coverage','',f'{sum(x["complete"] for x in coverage)} completed settings and {len(coverage)-sum(x["complete"] for x in coverage)} partial settings were observed. Partial costs are retained and flagged, but incomplete three-seed comparisons are excluded from bootstrap analysis. Re-run this script after active experiments complete.','',
       '## Statistical analysis','',
       'paired_objective_bootstrap.csv reports SAC-minus-control differences and 95% percentile intervals from 5,000 bootstrap draws (seed 20261101). Resampling is paired by objective ID, keeping its three evaluation-seed observations together. This avoids treating 150 repeated observations as 150 independent objectives. Intervals describe objective variation conditional on these fixed checkpoints and evaluation seeds; they do not quantify training-seed uncertainty. They are exploratory, unadjusted for multiple comparisons, and not evidence of causal superiority. Q/S is averaged across per-seed ratios; intervals are omitted if any bootstrap ratio is undefined. At Q=1, Q/S is redundant with ASR.','',
       '## Cost accounting','',
       'evaluation_costs.csv separates target, rewriter, and judge requests, observed input/output tokens, and summed request latencies. These sums are elapsed request time, not measured GPU-hours, energy, monetary charges, or clean isolated wall-clock benchmarks. Failed requests and incomplete settings are explicitly flagged. Output counts include any thinking tokens returned by the backend. Output-at-limit is a flag, not proof of truncation. Training costs are separately qualified in training_cost_evidence.csv; timestamp spans must not be called GPU training time.','',
       '## Policy analysis','',
       'policy_operator_frequencies.csv describes realized action concentration. representative_traces.csv uses the first completed success and failure in file order for each setting, shows operator and judge-score sequences, and cites raw traces. No actionable harmful prompt/response text is reproduced. These are judge-confirmed outcomes, not human-validated attack success.','',
       '## Dataset analysis','',
       'dataset_summary.csv and dataset_categories.csv describe the 440-item training snapshot, actual 50 full JBB evaluation objectives, and the available JailbreakHub source collection. The full source collection is not claimed to be the historical transfer subset. dataset_exact_overlap.csv checks whitespace-normalized, case-insensitive exact matches only; absence of exact overlap does not establish semantic disjointness.','',
       '## Remaining interpretation issues','',
       'Existing C1 checkpoints have terminal one-step replay and constant Success inputs; their multi-attempt evaluations do not establish learned sequential credit assignment. Action saturation and the available actor entropy-sign issue are documented in C1/diagnostic_audit. Statistical intervals and sensitivity analysis do not resolve these implementation/provenance issues or the historical label-input problem.','',
       '## Files','']
    report += [f'- {p.name}' for p in sorted(OUT.glob('*.csv'))]
    (OUT/'README.md').write_text('\n'.join(report)+'\n')
    print('Wrote',OUT)
    print('Dataset summary:',json.dumps(datasets))
    print('Exact overlaps:',json.dumps(overlaps))
    for row in statistics:
        print(row['study'],row['target'],row['contrast'],'ASR difference',round(row['ASR_difference'],4),
              '95% interval',round(row['ASR_CI_low'],4),round(row['ASR_CI_high'],4))

if __name__=='__main__':main()
