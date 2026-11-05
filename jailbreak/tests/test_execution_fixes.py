"""Offline behavioral regressions for goal extraction, stopping and aggregation."""
import csv
import importlib
import io
import json
import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from jailbreak.src.data.loader import export_jbb_harmful_to_seeds_ood, JBB_GOAL_SEEDS_FILENAME
from jailbreak.src.baselines.gcg import attack as gcg
from jailbreak.src.baselines.gcg.config import load_gcg_config
from jailbreak.src.baselines.autodan import attack as autodan
from jailbreak.src.baselines.autodan.config import load_autodan_config
from jailbreak.src.baselines.GPTFUZZER import attack as fuzz
from jailbreak.src.baselines.GPTFUZZER.config import load_gptfuzzer_config

ROOT = Path(__file__).resolve().parents[1]


class GoalExportTests(unittest.TestCase):
    def export(self, content, existing=None):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp);(root/'data').mkdir()
            (root/'data/harmful-behaviors.csv').write_text(content)
            dest = root/'new/seeds.jsonl'
            if existing is not None:
                dest.parent.mkdir();dest.write_text(existing)
            count = export_jbb_harmful_to_seeds_ood(root,dest)
            self.assertEqual(export_jbb_harmful_to_seeds_ood(root,dest),count)
            return count,[json.loads(l) for l in dest.read_text().splitlines()]

    def test_goal_wins_over_label_and_prompt(self):
        n,rows=self.export('Goal,Behavior,Prompt,Category\nFull fixture request,topic label,other text,fixture\n')
        self.assertEqual(n,1);self.assertEqual(rows[0]['text'],'Full fixture request')
        self.assertEqual(rows[0]['behavior'],'topic label');self.assertEqual(rows[0]['text_source'],'Goal')
        self.assertEqual(rows[0]['question_id'],'jbb_0')

    def test_missing_or_empty_goal_is_rejected(self):
        for content in ['Behavior\nlabel\n','Goal,Behavior\n,label\n','Goal,Behavior\n"   ",label\n']:
            with self.subTest(content=content),self.assertRaises(ValueError):self.export(content)

    def test_existing_different_snapshot_is_preserved(self):
        with self.assertRaises(FileExistsError):
            self.export('Goal,Behavior\nFull fixture request,label\n',existing='historical snapshot')

    def test_all_baseline_configs_use_versioned_full_goal_file(self):
        for method,loader_name in [('gcg','load_gcg_config'),('autodan','load_autodan_config'),('GPTFUZZER','load_gptfuzzer_config'),('PAIR','load_pair_config'),('TAP','load_tap_config'),('rlbreaker','load_rlbreaker_config'),('xjailbreak','load_xjailbreak_config')]:
            with self.subTest(method=method):
                module=importlib.import_module(f'jailbreak.src.baselines.{method}.config')
                config=getattr(module,loader_name)(ROOT/f'config/baselines/{method.lower()}.yaml')
                dataset = (config.datasets.by_key['test_jbb_ood'] if method == 'xjailbreak'
                           else config.datasets.test_jbb_ood)
                self.assertEqual(dataset.seeds_path.resolve(),(ROOT/'data/jbb/data'/JBB_GOAL_SEEDS_FILENAME).resolve())

    def test_frozen_sac_uses_identical_snapshot(self):
        from jailbreak.scripts.eval_old_sac_checkpoints import _jbb_goal_seeds
        self.assertEqual(_jbb_goal_seeds().resolve(),(ROOT/'data/jbb/data'/JBB_GOAL_SEEDS_FILENAME).resolve())


class StopOnSuccessTests(unittest.TestCase):
    def run_search(self,module,success_at,budget=5):
        if module is gcg:
            cfg=load_gcg_config(ROOT/'config/baselines/gcg.yaml');hp=cfg.gcg
            hp.num_steps=10;hp.candidates_per_step=2;hp.early_stop_improvement=100
        else:
            cfg=load_autodan_config(ROOT/'config/baselines/autodan.yaml');hp=cfg.autodan
            hp.num_generations=10;hp.population_size=2
        n=0
        def judge(*args,**kwargs):
            nonlocal n;n+=1;success=n==success_at
            return module.JudgeResult({'Success':.9 if success else .1},success,'{}')
        args=dict(seed_id='fixture',seed_text='Fixture task',target_llm=object(),judge_llm=object(),cfg=cfg,query_budget=budget,rng=random.Random(1),target_id='fixture-model',target_short_name='fixture',repeat_idx=0)
        with patch.object(module,'_call_target',return_value='Fixture response') as target,patch.object(module,'_score_with_judge',side_effect=judge),patch.object(module,'_compute_objective',side_effect=lambda *a: 10.0 if n==1 else 1.0):
            if module is gcg:
                result=module._optimize_suffix_for_pair(**args,gcg_cfg=hp,dataset_name='fixture')
            else:result=module._optimize_suffix_ga_for_pair(**args,ad_cfg=hp)
        expected=success_at if success_at is not None else budget
        self.assertEqual(target.call_count,expected)
        self.assertEqual(result[4],expected);self.assertEqual(len(result[3]),expected)
        self.assertEqual(result[2],success_at is not None)
        if expected:
            self.assertEqual(result[3][-1]['stop_reason'],'success' if success_at is not None else 'budget_exhausted')
            if success_at is not None:self.assertEqual(result[1]['Success'],.9)
        # JSONL and CSV schemas must both accept the new stopping metadata.
        fields=module._csv_fieldnames(cfg.judge.scoring.aspects) if module is gcg else module._csv_fieldnames()
        writer=csv.DictWriter(io.StringIO(),fieldnames=fields);writer.writeheader();writer.writerows(result[3])

    def test_stop_at_first_middle_or_last_allowed_attempt(self):
        for module in (gcg,autodan):
            for success_at in (1,2,5):
                with self.subTest(method=module.__name__,success_at=success_at):self.run_search(module,success_at)

    def test_failures_respect_cap_and_zero_budget(self):
        for module in (gcg,autodan):
            for budget in (0,1,5):
                with self.subTest(method=module.__name__,budget=budget):self.run_search(module,None,budget)


class GPTFuzzerSuccessTests(unittest.TestCase):
    def run_fuzz(self,init_prompts,outcomes,budget,early_stop,deduplicate=False):
        cfg=load_gptfuzzer_config(ROOT/'config/baselines/gptfuzzer.yaml');hp=cfg.gptfuzzer
        hp.population.deduplicate=deduplicate;hp.objective.early_stop_on_success=early_stop;hp.max_rounds=10
        fixtures=[(fuzz.JudgeResult({'Success':.9 if success else .1},success,'{}'),obj,'Fixture response') for success,obj in outcomes]
        with patch.object(fuzz,'_init_population',return_value=init_prompts),patch.object(fuzz,'_eval_candidate',side_effect=fixtures) as evaluate,patch.object(fuzz,'_sample_mutation_depth',return_value=1),patch.object(fuzz,'_sample_operator',return_value={'name':'fixture'}),patch.object(fuzz,'_apply_operator',return_value='Fixture child'):
            result=fuzz._fuzz_one_seed(seed_id='fixture',seed_text='Fixture task',target_llm=object(),judge_llm=object(),mutator_llm=None,cfg=cfg,hp=hp,query_budget=budget,rng=random.Random(1),target_id='fixture',target_short_name='fixture',repeat_idx=0,dataset_name='fixture')
        writer=csv.DictWriter(io.StringIO(),fieldnames=fuzz._csv_fieldnames());writer.writeheader();writer.writerows(result[3])
        return result,evaluate.call_count

    def test_initialization_stops_before_other_initial_candidates(self):
        result,n=self.run_fuzz(['first','second'],[(True,1)],5,True)
        self.assertEqual(n,1);self.assertTrue(result[2]);self.assertEqual(result[3][-1]['stop_reason'],'success')

    def test_lower_fitness_success_still_stops_and_counts(self):
        result,n=self.run_fuzz(['first'],[(False,10),(True,1)],5,True)
        self.assertEqual(n,2);self.assertTrue(result[2]);self.assertEqual(result[1]['Success'],.9)

    def test_any_success_survives_best_fitness_selection_without_early_stop(self):
        result,n=self.run_fuzz(['first','second'],[(False,10),(True,1),(False,20)],3,False)
        self.assertEqual(n,3);self.assertTrue(result[2])
        self.assertEqual(result[1]['Success'],.1) # best fitness is separate from success
        self.assertEqual(result[3][-1]['stop_reason'],'budget_exhausted')

    def test_success_from_duplicate_initial_prompt_is_not_lost(self):
        # Reproduces the historical DeepSeek discrepancy: a later response to
        # a duplicate initial prompt succeeded but was discarded from corpus.
        result,n=self.run_fuzz(['same','same'],[(False,0),(True,1)],2,False,deduplicate=True)
        self.assertEqual(n,2);self.assertTrue(result[2])
        self.assertEqual(result[1]['Success'],.1)

    def test_no_success_and_zero_budget(self):
        for budget,outcomes in [(0,[]),(1,[(False,1)])]:
            result,n=self.run_fuzz(['first'],outcomes,budget,True)
            self.assertFalse(result[2]);self.assertEqual(n,budget)


if __name__=='__main__':unittest.main()
