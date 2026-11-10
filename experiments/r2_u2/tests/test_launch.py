"""Pure prospective-contract tests; no research data or torch required."""
import copy
import ast
import contextlib
import io
import json
import tempfile
import types
from unittest.mock import Mock
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from acceptance import validate_events, verify_software
import launch


class LaunchLifecycleTests(unittest.TestCase):
    """Execute the actual launcher lifecycle with no data/model/GPU preflight.

    Slice main at fresh-directory allocation; everything after that remains the
    production code. A fixture stops at fit(), so no training is executed.
    """
    def exercise(self, root, fail_init=False):
        tree = ast.parse(Path(launch.__file__).read_text())
        main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        start = next(i for i, n in enumerate(main.body) if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == 'run' for t in n.targets))
        main.body = main.body[start:]
        module = ast.fix_missing_locations(ast.Module(body=[main], type_ignores=[]))
        run = root / 'experiments/r2_u2/runs/option_b_fixture'
        observations = []

        def initialize(*args, **kwargs):
            self.assertEqual(list(run.iterdir()), [])
            observations.append('empty_at_init')
            if fail_init:
                raise RuntimeError('fixture initialization failed')
            scaler = types.SimpleNamespace(get_scale=lambda: 65536, get_growth_interval=lambda: 2000,
                                           get_backoff_factor=lambda: .5, get_growth_factor=lambda: 2)
            def fit():
                self.assertEqual(json.loads((run / 'contract.json').read_text()), {'fixture': True})
                self.assertTrue((run / 'initial_baseline.json').is_file())
                observations.append('metadata_before_fit')
                raise RuntimeError('fixture stops before training')
            return types.SimpleNamespace(scaler=scaler, fit=fit)

        model = Mock()
        model.digests.return_value = {'fixture': 'initial'}
        scope = Mock()
        scope.frozen_state_digest.return_value = 'frozen_fixture'
        env = dict(vars(launch), data=root, uuid=types.SimpleNamespace(uuid4=lambda: types.SimpleNamespace(hex='fixture')),
                   scope=scope, stage=root, plan=types.SimpleNamespace(train_rows=[], selection_rows=[], selection_chunks=[]),
                   CropDataset=Mock(), ChunkDataset=Mock(), TrainConfig=lambda **kw: types.SimpleNamespace(**kw),
                   NumericalTrainer=initialize, model=model, contract={'fixture': True}, identity={'fixture': True},
                   recipe={'train': {'lr': .0001, 'weight_decay': .01}})
        exec(compile(module, launch.__file__, 'exec'), env)
        with contextlib.redirect_stdout(io.StringIO()):
            status = env['main']()
        return run, observations, status

    def test_empty_initialization_then_metadata_before_fit(self):
        with tempfile.TemporaryDirectory() as tmp:
            run, observations, status = self.exercise(Path(tmp))
            self.assertEqual(observations, ['empty_at_init', 'metadata_before_fit'])
            self.assertEqual(status, 1)  # deliberate fixture stop, never training
            result = json.loads((run / 'experiment_result.json').read_text())
            self.assertIn('fixture stops before training', result['traceback'])

    def test_initialization_failure_is_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run, observations, status = self.exercise(root, fail_init=True)
            self.assertEqual(observations, ['empty_at_init'])
            self.assertEqual(status, 1)
            result = json.loads((run / 'experiment_result.json').read_text())
            self.assertEqual(result['status'], 'experiment_failed')
            self.assertFalse(result['research_accepted'])
            self.assertIn('fixture initialization failed', result['traceback'])
            self.assertEqual((run / 'experiment_result.json').read_bytes(),
                             (root / 'experiments/r2_u2/reports/option_b_fixture/experiment_result.json').read_bytes())

    def test_existing_run_rejected_without_overwriting_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run, _, _ = self.exercise(root)
            before = {p.name: p.read_bytes() for p in run.iterdir()}
            with self.assertRaises(FileExistsError):
                self.exercise(root)
            self.assertEqual(before, {p.name: p.read_bytes() for p in run.iterdir()})


def events():
    out = []
    scale = 65536.
    for epoch in (1, 2):
        for skip in (False, True):
            after = scale * .5 if skip else scale
            out.append({'attempt': len(out) + 1, 'epoch': epoch,
                        'microbatch_sizes': [2] if skip else [4, 4, 4, 4],
                        'denominator': 2 if skip else 16, 'scale_before': scale, 'scale_after': after,
                        'applied': not skip, 'scaler_skipped': skip, 'norm_only_overflow': False,
                        'nonfinite_scaled_elements': skip, 'nonfinite_unscaled_elements': skip,
                        'preclip_norm': None if skip else 3., 'optimizer_step_calls_observed': 0 if skip else 1,
                        'scheduler_advanced': False, 'lr_before': [.0001], 'lr_after': [.0001]})
            scale = after
    return out


class ProspectiveTests(unittest.TestCase):
    def test_documented_skips_without_arbitrary_threshold(self):
        summary = validate_events(events(), records=18)
        self.assertEqual([s['scaler_skipped'] for s in summary], [1, 1])

    def test_partial_denominator_cannot_change(self):
        bad = events(); bad[1]['denominator'] = 16
        with self.assertRaises(ValueError): validate_events(bad, records=18)

    def test_unsafe_or_unexplained_updates_rejected(self):
        for field, value in [('optimizer_step_calls_observed', 0), ('norm_only_overflow', True), ('preclip_norm', float('inf'))]:
            bad = events(); bad[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError): validate_events(bad, records=18)

    def test_stale_or_skipped_software_gate_rejected(self):
        report = {'software_gate_passed': True, 'returncode': 0, 'test_cases': 17,
                  'counts': {'failure': 0, 'error': 0, 'skipped': 0}, 'verification_inputs_unchanged': True,
                  'expected_source_hashes': {'file': 'abc'}}
        report['command'] = ['python', '-B', '-m', 'pytest', 'experiments/r2_u2/tests/test_contract.py',
                             'experiments/r2_u2/tests/test_numerical.py', 'experiments/r2_u2/tests/test_launch.py',
                             '-q', '-ra', '-p', 'no:cacheprovider', '--basetemp', '/tmp/fixture', '--junitxml=/reports/junit.xml']
        self.assertTrue(verify_software(report, {'file': 'abc'}, 17))
        with self.assertRaises(ValueError): verify_software(report, {'file': 'changed'}, 17)
        bad = copy.deepcopy(report); bad['counts']['skipped'] = 1
        with self.assertRaises(ValueError): verify_software(bad, {'file': 'abc'}, 17)


if __name__ == '__main__': unittest.main()
