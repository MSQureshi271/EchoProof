import sys
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from r3_n1_ast_supplement import reliability, diagram


class ReliabilityTests(unittest.TestCase):
    def test_perfect_endpoints(self):
        bins,ece=reliability([0,1],[0.,1.])
        self.assertEqual(ece,0)
        self.assertEqual(bins[9]['count'],1)

    def test_fixed_bin_edges(self):
        bins,ece=reliability([0,1],[.1,.9])
        self.assertEqual(bins[1]['count'],1)
        self.assertEqual(bins[9]['count'],1)
        self.assertAlmostEqual(ece,.1)

    def test_weighting_and_empty_bins(self):
        bins,ece=reliability([0,1,1],[.2,.2,.8])
        self.assertAlmostEqual(ece,(2*.3+.2)/3)
        self.assertIsNone(bins[0]['mean_score'])

    def test_diagram_valid_svg(self):
        bins,_=reliability([0,1],[.1,.9])
        root=ET.fromstring(diagram({'raw':bins,'temperature':bins}))
        self.assertTrue(root.tag.endswith('svg'))

if __name__=='__main__':
    unittest.main()
