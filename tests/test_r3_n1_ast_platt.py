import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import soundfile as sf
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from echoproof.models.r3_n1_ast_platt_bundle import platt_score,inspect_ast_platt_candidate,load_ast_platt_candidate
from echoproof.bundle import write_bundle,BundleError
from echoproof.contracts import BundleSpecification,DecisionRule
from echoproof.data.chunks import PREPROCESSING_HASH


class PlattTests(unittest.TestCase):
    def test_identity_and_clipping(self):
        for p in [0.,1e-9,.2,.5,.9,1.]:
            self.assertAlmostEqual(platt_score(p,1,0),min(max(p,1e-7),1-1e-7),places=14)

    def test_invalid_rejected(self):
        for p,a,b in [(float('nan'),1,0),(.5,0,0),(.5,-1,0),(.5,1,float('inf')),(-.1,1,0),(True,1,0)]:
            with self.assertRaises(ValueError):platt_score(p,a,b)

    def test_monotone_and_extreme(self):
        z=[platt_score(p,1.391,1.097) for p in np.linspace(0,1,100)]
        self.assertTrue(all(a<b for a,b in zip(z,z[1:])))
        self.assertEqual(platt_score(.9,1e5,1e5),1.)

    def test_aggregate_before_calibration(self):
        self.assertNotAlmostEqual(platt_score(.35,1.4,1.1),np.mean([platt_score(.1,1.4,1.1),platt_score(.6,1.4,1.1)]))

    def make_bundle(self,path):
        rule=DecisionRule(1,'threshold',.63)
        spec=BundleSpecification(1,'bundle:test-platt','ast_partial:B',PREPROCESSING_HASH,rule,'genuine','tts')
        meta={'schema':'ast_platt_candidate_v1','method':'platt_on_recording_log_odds','a':1.4,'b':1.1,'epsilon':1e-7,
              'checkpoint_sha256':'a'*64,'decision_rule':rule.to_dict(),'reference_precision':'fp16','officially_promoted':False}
        components={n:b'{}' for n in ['model/config.json','model/preprocessor_config.json','model/model.safetensors']}
        components['model/ast_detector.json']=json.dumps({'source_checkpoint_sha256':'a'*64}).encode()
        components['calibration/platt.json']=json.dumps(meta).encode()
        write_bundle(path,spec,components)

    def test_roundtrip_and_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'bundle';self.make_bundle(path)
            b,m=inspect_ast_platt_candidate(path)
            self.assertEqual(m['a'],1.4)
            self.assertEqual(b.spec.decision_rule.decide(.63),1)
            (path/'calibration/platt.json').write_text('{}')
            with self.assertRaises(BundleError):inspect_ast_platt_candidate(path)

    def test_precision_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'bundle';self.make_bundle(path)
            with self.assertRaises(ValueError):load_ast_platt_candidate(path,device='cpu',precision='fp32')

    def test_actual_inference_pipeline_with_fixture_scorer(self):
        # Real decode/chunk/aggregate/calibrate/decision; mocked AST, not model parity.
        from echoproof.inference import analyze
        class Fixture:
            def score_chunks(self,waves):return np.linspace(.1,.6,len(waves))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'bundle';self.make_bundle(path)
            with patch('echoproof.models.ast_adapter.ASTBundleAdapter.load',return_value=Fixture()):
                bundle=load_ast_platt_candidate(path)
            wave=(.1*np.sin(2*np.pi*440*np.arange(96000)/16000)).astype('float32')
            audio=io.BytesIO();sf.write(audio,wave,16000,format='WAV',subtype='PCM_16')
            result=analyze(audio.getvalue(),bundle)
            self.assertEqual(result.result.status,'success')
            expected=platt_score(float(np.mean(result.chunk_scores)),1.4,1.1)
            self.assertAlmostEqual(result.result.score,expected)


if __name__=='__main__':unittest.main()
