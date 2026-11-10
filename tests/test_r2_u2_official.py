"""Official R2-U2 integration contracts; no torch, data, or training required."""
import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class OfficialR2U2Tests(unittest.TestCase):
    def test_official_entry_points_delegate_to_pinned_sources(self):
        train = (ROOT / 'scripts/r2_u2_ast_warmup.py').read_text()
        verify = (ROOT / 'scripts/r2_u2_verify.py').read_text()
        self.assertIn('experiments/r2_u2/scripts/launch.py', train)
        self.assertIn('experiments/r2_u2/acceptance/verify_handoff.py', verify)

    def test_official_index_binds_accepted_initializer(self):
        cfg = json.loads((ROOT / 'configs/r2_u2_ast_b.json').read_text())
        checkpoint = ROOT / cfg['initializer']
        self.assertTrue(checkpoint.is_file(), f'copy accepted checkpoint from Drive to {checkpoint}')
        self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(), cfg['initializer_sha256'])
        self.assertEqual(cfg['run_id'], 'option_b_1b3757e898da4baaad2bde49fa3cea01')

    def test_r3_handoff_is_weights_only_and_does_not_change_shared_trainer(self):
        doc = (ROOT / 'docs/r2_u2_acceptance.md').read_text()
        self.assertIn('loading **only** checkpoint `model` weights', doc)
        self.assertIn('new run and optimizer', doc)
        self.assertIn('Shared Trainer', doc)
        self.assertIn('recipient', doc.lower())


if __name__ == '__main__':
    unittest.main()
