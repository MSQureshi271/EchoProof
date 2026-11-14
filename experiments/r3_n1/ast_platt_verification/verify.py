"""Colab-only forward verification; no fitting, updates, publication or lock writes."""
import argparse, hashlib, io, json, platform, sys, uuid
from pathlib import Path
from datetime import datetime,timezone

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'source/src'))

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main(root,candidate):
    import numpy as np
    import pyarrow.parquet as pq
    import soundfile as sf
    import torch,transformers
    from echoproof.models.r3_n1_ast_platt_bundle import load_ast_platt_candidate,platt_score
    from echoproof.inference import analyze
    from echoproof.training.datasets import ShardWaveformLoader
    from echoproof.training.r3_m1_data import VerifiedWaveformLoader
    from echoproof.training.r3_u2_parity import ATOL_FP16
    root,candidate=Path(root).resolve(),Path(candidate).resolve()
    manifest=json.loads((HERE/'source_manifest.json').read_text())
    for rel,expected in manifest.items():
        if sha(HERE/rel)!=expected:raise ValueError('Verification source mismatch: '+rel)
    if not torch.cuda.is_available():raise RuntimeError('Select the existing Colab T4 GPU runtime')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    originals={p:sha(p) for p in candidate.rglob('*') if p.is_file()}
    for rel,expected in json.loads((candidate/'manifest.json').read_text()).items():
        if sha(candidate/rel)!=expected:raise ValueError('Candidate mismatch: '+rel)
    prospective=json.loads((candidate/'prospective.json').read_text())
    for rel,expected in prospective['sources'].items():
        if rel.startswith('src/') and sha(HERE/'source'/rel)!=expected:
            raise ValueError('Source differs from candidate provenance: '+rel)
    inputs=['data/manifests/cohort_v1.parquet','data/splits/protocol_B_v1.parquet','data/processed/chunks_v1.parquet',
            'runs/r3/r3_u2_20261112T134049Z/predictions_chunks_operating.parquet',
            'runs/r3/r3_u2_20261112T134049Z/predictions_recordings_operating.parquet']
    for rel in inputs:
        if sha(root/rel)!=prospective['inputs'][rel]:raise ValueError('Input mismatch: '+rel)
        originals[root/rel]=sha(root/rel)
    out=root/'reports/r3/ast_platt_recipient_verification'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8])
    out.mkdir(parents=True,exist_ok=False)
    report={'status':'started','candidate':str(candidate),'manifest_sha256':sha(candidate/'manifest.json'),
            'source_manifest_sha256':sha(HERE/'source_manifest.json'),'gpu':torch.cuda.get_device_name(0),
            'torch':torch.__version__,'transformers':transformers.__version__,'python':platform.python_version(),
            'training':False,'official_promotion':False,'precision':'fp16','records':[]}
    try:
        bundle=load_ast_platt_candidate(candidate/'bundle',device='cuda',precision='fp16')
        records=pq.read_table(candidate/'operating_predictions.parquet').to_pylist()
        # Four nearest decision-boundary recordings + two fixed-ID-hash samples/component.
        chosen=sorted(records,key=lambda r:(abs(r['platt_score']-bundle.spec.decision_rule.threshold),r['recording_id']))[:4]
        for component in ['genuine','melo','vits','xtts']:
            chosen+=sorted([r for r in records if r['component']==component],key=lambda r:hashlib.sha256(r['recording_id'].encode()).hexdigest())[:2]
        chosen=list({r['recording_id']:r for r in chosen}.values())
        cohort=pq.read_table(root/'data/manifests/cohort_v1.parquet').to_pylist()
        by_id={r['recording_id']:r for r in cohort}
        split=[json.loads(r['payload_json']) for r in pq.read_table(root/'data/splits/protocol_B_v1.parquet').to_pylist()]
        allowed={r['recording_id'] for r in split if r['partition']=='validation' and r['validation_role']=='operating'}
        if not {r['recording_id'] for r in chosen}<=allowed:raise ValueError('Non-operating recording selected')
        selected=[by_id[r['recording_id']] for r in chosen]
        base=ShardWaveformLoader(root/'data/processed/cohort_v1',{r['recording_id']:r['canonical_path'] for r in selected})
        loader=VerifiedWaveformLoader(base,selected)
        chunks=[json.loads(r['payload_json']) for r in pq.read_table(root/'data/processed/chunks_v1.parquet').to_pylist()]
        exports={r['chunk_id']:r['tts_probability'] for r in pq.read_table(root/inputs[3]).to_pylist()}
        report['selection_rule']='4 nearest calibrated threshold plus 2 fixed-ID-hash/component; deduplicated; no outcome-driven reselection'
        try:
            for row in chosen:
                rid=row['recording_id'];wave=loader(rid)
                data=io.BytesIO();sf.write(data,wave,16000,format='WAV',subtype='PCM_16')
                result=analyze(data.getvalue(),bundle)
                if result.result.status!='success':raise ValueError(f'Inference failed for {rid}: {result.result}')
                expected=sorted([c for c in chunks if c['protocol_id']=='protocol:B' and c['recording_id']==rid and c['partition']=='validation'],key=lambda c:c['start_sample'])
                if list(result.windows)!=[(c['start_sample'],c['end_sample']) for c in expected]:raise ValueError('Canonical windows differ: '+rid)
                diff=max(abs(a-exports[c['chunk_id']]) for a,c in zip(result.chunk_scores,expected))
                decision=bundle.spec.decision_rule.decide(float(result.result.score))
                order=platt_score(float(np.mean(result.chunk_scores)),json.loads((candidate/'fit.json').read_text())['a'],json.loads((candidate/'fit.json').read_text())['b'])
                report['records'].append({'recording_id':rid,'component':row['component'],'chunks':len(expected),
                    'max_chunk_difference':diff,'raw_recording_difference':abs(result.raw_score-row['raw_score']),
                    'platt_difference':abs(result.result.score-row['platt_score']),'decision':decision,'expected_decision':row['decision'],
                    'aggregation_calibration_order_matches':abs(order-result.result.score)<=1e-15})
        finally:loader.close()
        report['inherited_chunk_tolerance']=ATOL_FP16
        report['chunk_parity_passed']=all(r['max_chunk_difference']<=ATOL_FP16 for r in report['records'])
        report['all_sampled_decisions_match']=all(r['decision']==r['expected_decision'] for r in report['records'])
        report['aggregation_order_passed']=all(r['aggregation_calibration_order_matches'] for r in report['records'])
        report['status']='sample_checks_passed_review_pending' if all(report[k] for k in ['chunk_parity_passed','all_sampled_decisions_match','aggregation_order_passed']) else 'discrepancy_requires_review'
        report['limitations']=['Sample only, not all operating recordings','CPU/fp32 not verified','No automatic approval; inherited chunk tolerance is not a bound on all recording scores']
    except Exception as exc:
        report.update(status='verification_failed',error_type=type(exc).__name__,error=str(exc))
    finally:
        report['originals_unchanged']=all(sha(p)==v for p,v in originals.items())
        if not report['originals_unchanged']:report['status']='verification_failed_inputs_changed'
        (out/'verification.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        print(out/'verification.json',flush=True);print(json.dumps(report,indent=2))
    if report['status']!='sample_checks_passed_review_pending':raise SystemExit(1)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project-root',required=True);p.add_argument('--candidate',required=True)
    a=p.parse_args();main(a.project_root,a.candidate)
