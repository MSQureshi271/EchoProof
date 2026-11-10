"""Official copies must retain the accepted numerical contract."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from echoproof.training.r2_u2_contract import verify_official_copies, provenance

class OfficialTests(unittest.TestCase):
    def test_accepted_copy_equivalence(self):
        verify_official_copies()
        self.assertEqual((ROOT / 'configs/r2_u2_ast_b.json').read_bytes(),
                         (ROOT / 'experiments/r2_u2/configs/historical_recipe.json').read_bytes())

    def test_divergence_cannot_inherit_accepted_provenance(self):
        read = Path.read_text
        def changed(path, *args, **kwargs):
            value = read(path, *args, **kwargs)
            return value + 'changed' if path.name == 'r2_u2_numerical_trainer.py' else value
        with patch.object(Path, 'read_text', changed), self.assertRaises(ValueError):
            provenance({'train': {}}, {'fixture': True}, fixture=True)

    def test_fixture_identity(self):
        value = provenance({'train': {'seed': 42}}, {'fixture': 'tiny'}, fixture=True)
        self.assertTrue(value['fixture_only'])
        self.assertFalse(value['research_accepted'])

if __name__ == '__main__': unittest.main()
