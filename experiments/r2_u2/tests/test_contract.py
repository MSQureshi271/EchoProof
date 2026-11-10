"""No scientific packages required."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from contract import provenance, sources, fingerprint


class ContractTests(unittest.TestCase):
    def test_fixture_identity_binds_sources_recipe_model(self):
        a = provenance({'train': {'seed': 42}}, {'fixture': 'tiny'}, fixture=True)
        self.assertEqual(a['sources'], sources())
        self.assertFalse(a['research_accepted'])
        b = provenance({'train': {'seed': 43}}, {'fixture': 'tiny'}, fixture=True)
        self.assertNotEqual(fingerprint(a), fingerprint(b))

    def test_strict_json_rejects_nonfinite(self):
        with self.assertRaises(ValueError):
            fingerprint({'norm': float('inf')})


if __name__ == '__main__':
    unittest.main()
