#!/usr/bin/env python3
"""Reconcile recorded objective success without modifying any original result."""
import argparse
import collections
import hashlib
import json
from pathlib import Path


def reconcile(run_dir, hp):
    from jailbreak.src.baselines.GPTFUZZER.attack import _dedup_key
    metric_path = run_dir/'metrics.json'
    event_path = run_dir/'events.jsonl'
    stored = json.loads(metric_path.read_text())
    snapshot_path = run_dir/'config.snapshot.json'
    snapshot = json.loads(snapshot_path.read_text())
    population = snapshot['gptfuzzer']['population']
    for key in ('deduplicate', 'dedup_key', 'max_population'):
        if getattr(hp.population, key) != population[key]:
            raise ValueError(f"Historical population setting mismatch: {key}")
    groups = collections.defaultdict(list)
    for line in event_path.open():
        if line.strip():
            event = json.loads(line)
            groups[str(event['seed_id'])].append(event)
    affected = []; replayed = 0; successes = 0; calls = 0
    for sid, events in groups.items():
        corpus = []; seen = set(); best = None; discarded = []
        for event in events:
            key = _dedup_key(event['candidate_prompt'], hp)
            if hp.population.deduplicate and key in seen:
                discarded.append(event)
            else:
                seen.add(key); corpus.append(event)
                corpus.sort(key=lambda row: row['objective'], reverse=True)
                corpus = corpus[:max(1, hp.population.max_population)]
            if event['step'] == 0:
                best = corpus[0]
            elif event['objective'] > best['objective']:
                best = event
        legacy = bool(best and best['success'])
        any_success = any(event['success'] for event in events)
        replayed += legacy; successes += any_success; calls += len(events)
        if any_success != legacy:
            affected.append({
                'seed_id': sid,
                'legacy_best_success': legacy,
                'any_attempt_success': any_success,
                'successful_discarded_initial_attempts': [
                    {'query': e['queries_used'], 'Success': e['score_Success']}
                    for e in discarded if e['step'] == 0 and e['success']
                ],
            })
    return {
        'scope': 'Historical logs under their original inputs and judge; not a fresh corrected experiment.',
        'source_events': str(event_path.resolve()),
        'source_events_sha256': hashlib.sha256(event_path.read_bytes()).hexdigest(),
        'source_metrics_sha256': hashlib.sha256(metric_path.read_bytes()).hexdigest(),
        'population_settings_verified_against_snapshot': True,
        'N': len(groups), 'target_calls': calls,
        'stored_successes': stored['num_success'],
        'replayed_legacy_successes': replayed,
        'legacy_replay_matches_stored': replayed == stored['num_success'],
        'any_attempt_successes': successes,
        'any_attempt_ASR': successes/len(groups) if groups else None,
        'any_attempt_QS': calls/successes if successes else None,
        'affected_objectives': affected,
    }


def main():
    from jailbreak.src.baselines.GPTFUZZER.config import load_gptfuzzer_config
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True,
                        help='Config with the historical population/dedup settings.')
    parser.add_argument('--out', type=Path, required=True,
                        help='New report path; original files must not be overwritten.')
    args = parser.parse_args()
    cfg = load_gptfuzzer_config(args.config)
    report = reconcile(args.run_dir, cfg.gptfuzzer)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
