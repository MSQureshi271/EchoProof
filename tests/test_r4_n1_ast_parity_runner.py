"""Exercise the complete runner with fixture model outputs, without torch/GPU."""
import ast
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


class ParityRunnerTest(unittest.TestCase):
    def test_all_components_full_recordings_and_report(self):
        from echoproof.artifacts import digest
        from echoproof.training.r3_u2_parity import parity_check
        tree=ast.parse((ROOT/'src/echoproof/training/r4_u1_ast_data.py').read_text())
        fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='select_chunks')
        env={'FULL_CHUNK':64000,'digest':digest,'R4U1DataError':ValueError}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'<A sampler>','exec'),env)
        components=['genuine','melo','vits','xtts','chatterbox']
        chunks=[types.SimpleNamespace(chunk_id=f'c:{c}:{i}',recording_id=f'r:{c}:{i//4}',valid_samples=32000 if i%2 else 64000) for c in components for i in range(12)]
        byrid={c.recording_id:c.chunk_id.split(':')[1] for c in chunks}
        dataset=types.SimpleNamespace(chunks=chunks)
        selected=env['select_chunks'](dataset,byrid,50)
        self.assertEqual({byrid[chunks[i].recording_id] for i in selected},set(components))
        closed=[]
        data=types.ModuleType('echoproof.training.r4_u1_ast_data')
        data.select_chunks=env['select_chunks']
        data.build_data=lambda *a,**k:{'validation':{'selection':dataset},'validation_components':byrid,'loader':types.SimpleNamespace(close=lambda:closed.append(True))}
        helper=types.ModuleType('echoproof.training.r3_u2_parity')
        helper.parity_check=parity_check
        helper.waves_for=lambda ds,indices:[np.array([.1 if byrid[ds.chunks[i].recording_id]=='genuine' else .9]) for i in indices]
        adapter=types.ModuleType('echoproof.models.ast_adapter')
        adapter.ASTBundleAdapter=types.SimpleNamespace(load=lambda *a,**k:types.SimpleNamespace(score_chunks=lambda waves:np.array([w[0] for w in waves])))
        spec=importlib.util.spec_from_file_location('parity_runner_fixture',ROOT/'scripts/r4_n1_ast_parity.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        from echoproof.contracts import DecisionRule
        with tempfile.TemporaryDirectory() as tmp:
            report=Path(tmp)/'report.json'
            fake=types.SimpleNamespace(spec=types.SimpleNamespace(decision_rule=DecisionRule(1,'threshold',.5)),components={'calibration/ast_a.json':Path(tmp)/'meta'})
            expected=[{'chunk_id':c.chunk_id,'tts_probability':.1 if byrid[c.recording_id]=='genuine' else .9} for c in chunks]
            with patch.dict(sys.modules,{'echoproof.training.r4_u1_ast_data':data,'echoproof.training.r3_u2_parity':helper,'echoproof.models.ast_adapter':adapter}),patch.object(module,'inspect',return_value=(fake,{'method':'identity','parameters':{},'accepted':False})),patch.object(module,'digest',return_value='a'*64),patch('pyarrow.parquet.read_table',return_value=types.SimpleNamespace(to_pylist=lambda:expected)),patch.object(sys,'argv',['parity','--bundle',tmp,'--shard-dir',tmp,'--report',str(report)]):
                module.main()
            result=json.loads(report.read_text())
            self.assertTrue(result['passed']);self.assertEqual(result['decision_flips'],[])
            self.assertEqual(result['items'],60)
            self.assertEqual(len(result['comparisons']),15)
            self.assertEqual(closed,[True])
