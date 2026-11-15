import unittest
import tempfile
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from echoproof.evaluation.r4_n1_ast import accepted, mapping, validate_rows, loso
from echoproof.evaluation.threshold import select_threshold


class ASTACalibrationTests(unittest.TestCase):
    def test_strict_acceptance_and_identity_fallback(self):
        raw = {'log_loss':.3,'brier':.1}
        self.assertTrue(accepted('affine',{'ok':True},raw,{'log_loss':.2,'brier':.1}))
        self.assertFalse(accepted('affine',{'ok':True},raw,raw))
        self.assertFalse(accepted('identity',{'ok':True},raw,{'log_loss':.2,'brier':.09}))
        self.assertFalse(accepted('affine',{'ok':False},raw,{'log_loss':.2,'brier':.09}))

    def test_identity_and_temperature_one(self):
        p = [.1,.5,.9]
        np.testing.assert_allclose(mapping('temperature',{'temperature':1},p),p)
        np.testing.assert_array_equal(mapping('identity',{},p),p)

    def test_role_guard_rejects_test_and_invalid_scores(self):
        row = dict(recording_id='x',group_id='speaker:1',label=0,component='genuine',validation_role='calibration',raw_score=.2)
        expected = {'x':SimpleNamespace(group_id='speaker:1')}
        validate_rows([row],expected,'calibration',1)
        for change in ({'validation_role':'test'},{'label':1},{'raw_score':float('nan')}):
            with self.assertRaises(ValueError):
                validate_rows([{**row,**change}],expected,'calibration',1)

    def test_threshold_rule_is_saved_contract_rule(self):
        rule,f1,_ = select_threshold([0,0,1,1],[.1,.2,.8,.9])
        self.assertEqual(f1,1)
        self.assertEqual(rule.threshold,.8)

    def test_loso_identity_on_uninformative_scores(self):
        recipe=json.loads((Path(__file__).resolve().parents[1]/'configs/r4_n1_ast_calibration_a.json').read_text())
        result=loso(np.array([0,1]*3),np.array([.5]*6),np.array(['speaker:1']*2+['speaker:2']*2+['speaker:3']*2),recipe)
        self.assertEqual(result['selected'],'identity')

    def test_bundle_roundtrip_corruption_and_no_overwrite(self):
        from echoproof.bundle import write_bundle, load_bundle
        from echoproof.contracts import BundleSpecification, DecisionRule
        from echoproof.models.r4_n1_ast_bundle import inspect
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'bundle'
            rule=DecisionRule(1,'threshold',.5)
            spec=BundleSpecification(1,'bundle:test','ast_partial:A','a'*64,rule,'genuine','tts')
            meta={'checkpoint_sha256':'b'*64,'rule':rule.to_dict(),'score_type':'model_score','accepted':False,'method':'identity','parameters':{}}
            components={'model/ast_detector.json':json.dumps({'source_checkpoint_sha256':'b'*64}).encode(),'calibration/ast_a.json':json.dumps(meta).encode()}
            write_bundle(path,spec,components)
            inspect(path)
            with self.assertRaises(Exception):write_bundle(path,spec,components)
            (path/'calibration/ast_a.json').write_text('{}')
            with self.assertRaises(Exception):load_bundle(path)
