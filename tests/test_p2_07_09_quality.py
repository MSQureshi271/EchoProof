import hashlib
import json
import struct
import tempfile
import unittest
import wave
import types
import sys
from unittest.mock import patch
from pathlib import Path
from echoproof.data.phase2_quality import _riff_check, _source, capacity, decode_sample, inspect_header, reconcile, inspect
from echoproof.data.phonemedf_adapter import adapt_row


class QualityTest(unittest.TestCase):
    def test_truncated_riff_and_safe_paths(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'dataset').mkdir()
            p=root/'dataset/a.wav'
            p.write_bytes(b'RIFF'+struct.pack('<I',100)+b'WAVE')
            with self.assertRaisesRegex(ValueError,'truncated'): _riff_check(p)
            with self.assertRaises(ValueError): _source(root,'dataset/../outside')

    def test_capacity_unknown_and_reserve(self):
        with tempfile.TemporaryDirectory() as d:
            pending=capacity(d,scratch_path=d,planned_download_bytes=100,expanded_bytes=200)
            self.assertEqual(pending['drive_capacity_check'],'pending')
            self.assertEqual(pending['required_additional_bytes_with_reserve'],375)
            passed=capacity(d,scratch_path=d,drive_free_gb=1,remaining_hours=1)
            self.assertEqual(passed['acquisition'],'none_planned')
            with self.assertRaises(ValueError): capacity(d,scratch_path=d,planned_download_bytes=1)

    def test_reconciliation_propagates_attention_and_unrun_stages(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            def write(relative, value):
                path=root/relative;path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text(json.dumps(value)+'\n',encoding='utf8');return path
            def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
            inv=root/'data/manifests/phase2/i';inv.mkdir(parents=True)
            comps=('genuine','xtts','chatterbox','melo','vits')
            rows=[{'component':c,'size_bytes':2} for c in comps]+[{'component':'transcripts','size_bytes':1}]
            (inv/'file_inventory.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf8')
            (inv/'transcript_inventory.jsonl').write_text('{}\n',encoding='utf8')
            invsum=write('data/manifests/phase2/i/scan_summary.json',{'artifacts_sha256':{name:digest(inv/name) for name in ('file_inventory.jsonl','transcript_inventory.jsonl')}})
            snap=root/'data/manifests/phase2_raw/s';snap.mkdir(parents=True)
            mapping=snap/'mapping_evidence.jsonl';mapping.write_text(''.join(json.dumps({'component':c})+'\n' for c in comps),encoding='utf8')
            provenance=write('data/manifests/phase2_raw/s/provenance.json',{'input_inventory':'data/manifests/phase2/i','inventory_summary_sha256':digest(invsum)})
            sm=write('data/manifests/phase2_raw/s/snapshot_manifest.json',{'status':'ready_for_review','counts':{'recordings':5},'artifacts_sha256':{'mapping_evidence.jsonl':digest(mapping),'provenance.json':digest(provenance)}})
            pair=write('data/manifests/phase2_pairs/p/pairing_summary.json',{'status':'ready_for_review','snapshot_manifest_sha256':digest(sm),'artifacts_sha256':{},'transcript_diagnostics':{'ok':1},'audio_dispositions':5,'candidate_edges':5,'verified_edges':0})
            insp=root/'data/manifests/phase2_inspection/h';insp.mkdir(parents=True)
            sample=insp/'sample_decodes.jsonl';sample.write_text(json.dumps({'component':'genuine','state':'resource_limit'})+'\n',encoding='utf8')
            write('data/manifests/phase2_inspection/h/inspection_summary.json',{'status':'needs_attention','pairing_summary_sha256':digest(pair),'artifacts_sha256':{'sample_decodes.jsonl':digest(sample)},'component_header_counts':{c:{'ok':1} for c in comps},'sample_outcomes':{'resource_limit':1}})
            result=reconcile(root,{'outputs':{'inventory':'data/manifests/phase2'}},snap,pair.parent,insp)
            self.assertEqual(result['status'],'needs_attention')
            self.assertEqual(result['unique_genuine_controls'],1)
            self.assertEqual(result['stages']['partitioned']['state'],'not_run')

    def test_inspection_resume_uses_committed_header_rows(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);comps=('genuine','xtts','chatterbox','melo','vits')
            sources={c:'dataset/'+c for c in comps}
            config={'sources':sources,'source_registry':'configs/data_sources.json','sample_count':5,'sample_seed':42,'checkpoint_interval':1,'decode_limit_seconds':120}
            def write(path,obj):
                p=root/path;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj)+'\n',encoding='utf8');return p
            def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
            write('configs/data_sources.json',{'sources':[{'source_id':'phonemedf','publisher_revision_verified':True}]})
            entries=[]
            for comp in comps:
                r={'component':comp,'file_id':'phonemedf:'+comp,'state':'completed','file_type':'.wav','relative_path':'one.wav','sha256':'a'*64}
                rec,_=adapt_row(r,config,'phonemedf:snapshot')
                entries.append({'schema_name':'recording','schema_version':1,'payload_json':rec.to_json()})
            snap=write('data/manifests/phase2_raw/s/snapshot_manifest.json',{'status':'ready_for_review','counts':{'recordings':5},'artifacts_sha256':{}})
            pairdir=root/'data/manifests/phase2_pairs/p';pairdir.mkdir(parents=True)
            (pairdir/'coverage.jsonl').write_text(json.dumps({'candidate_stem':'one','counts':{k:1 for k in (*comps,'transcripts')}})+'\n',encoding='utf8')
            (pairdir/'pair_dispositions.jsonl').write_text(''.join(json.dumps({'recording_id':'phonemedf:'+c,'candidate_stem':'one','has_genuine':True,'has_transcript':True})+'\n' for c in comps),encoding='utf8')
            write('data/manifests/phase2_pairs/p/pairing_summary.json',{'status':'ready_for_review','snapshot_manifest_sha256':sha(snap),'artifacts_sha256':{n:sha(pairdir/n) for n in ('coverage.jsonl','pair_dispositions.jsonl')}})
            fakearrow=types.ModuleType('pyarrow');fakearrow.__version__='fixture';fakearrow.__path__=[]
            fakepq=types.ModuleType('pyarrow.parquet')
            class FakeFile:
                schema_arrow=types.SimpleNamespace(names=['schema_name','schema_version','payload_json'])
                def __init__(self,path): pass
                def iter_batches(self,batch_size): yield types.SimpleNamespace(to_pylist=lambda:entries)
            fakepq.ParquetFile=FakeFile
            fakesound=types.ModuleType('soundfile');fakesound.__version__='fixture'
            fakenumpy=types.ModuleType('numpy');fakenumpy.__version__='fixture'
            calls=[]
            def fake_header(path,record_id,component,relative):
                calls.append(record_id)
                if len(calls)==3: raise KeyboardInterrupt()
                return {'file_id':record_id,'component':component,'path':relative,'state':'ok','duration_seconds':5,'frames':5,'sample_rate':1,'channels':1}
            def fake_decode(path,header,max_seconds): return {'file_id':header['file_id'],'component':header['component'],'path':header['path'],'state':'ok'}
            with patch.dict(sys.modules,{'pyarrow':fakearrow,'pyarrow.parquet':fakepq,'soundfile':fakesound,'numpy':fakenumpy}), patch('echoproof.data.phase2_quality.inspect_header',side_effect=fake_header), patch('echoproof.data.phase2_quality.decode_sample',side_effect=fake_decode):
                with self.assertRaises(KeyboardInterrupt): inspect(root,config,snap.parent,pairdir)
                checkpoint=next((root/'data/manifests/phase2_inspection').iterdir())
                result=inspect(root,config,snap.parent,pairdir,resume=checkpoint.name)
            self.assertEqual(result['header_counts']['ok'],5)
            self.assertEqual(result['sample_count'],5)
            self.assertEqual(calls.count('phonemedf:genuine'),1)
            self.assertEqual(calls.count('phonemedf:xtts'),1)


try:
    import numpy as np
    import soundfile as sf
except ModuleNotFoundError:
    np=sf=None


@unittest.skipIf(sf is None, 'numpy and soundfile unavailable')
class AudioQualityTest(unittest.TestCase):
    def test_valid_stereo_and_resource_limit(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'ok.wav';sf.write(p,np.zeros((16000,2)),16000)
            header=inspect_header(p,'fixture:a','genuine','dataset/a.wav')
            self.assertEqual(header['state'],'ok')
            self.assertEqual(decode_sample(p,header,2)['state'],'ok')
            self.assertEqual(decode_sample(p,header,0.5)['state'],'resource_limit')

    def test_corrupt_and_nonfinite(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'corrupt.wav';p.write_bytes(b'RIFF'+struct.pack('<I',100)+b'WAVE')
            self.assertEqual(inspect_header(p,'fixture:x','genuine','dataset/x.wav')['state'],'header_error')
            p=Path(d)/'nonfinite.wav';sf.write(p,np.array([0.,float('nan'),0.],dtype=np.float32),16000,subtype='FLOAT')
            header=inspect_header(p,'fixture:y','genuine','dataset/y.wav')
            self.assertEqual(decode_sample(p,header,1)['state'],'decode_error')

if __name__=='__main__': unittest.main()
