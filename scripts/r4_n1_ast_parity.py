"""Recipient GPU verification for AST-A candidate; never trains."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from echoproof.evaluation.r4_n1_ast import digest, mapping, write
from echoproof.models.r4_n1_ast_bundle import inspect


def main():
    import pyarrow.parquet as pq
    from echoproof.training.r4_u1_ast_data import build_data, select_chunks
    from echoproof.training.r3_u2_parity import waves_for, parity_check
    from echoproof.models.ast_adapter import ASTBundleAdapter
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True)
    p.add_argument('--shard-dir',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    a=p.parse_args()
    bundle,meta=inspect(a.bundle)
    data=build_data(ROOT,a.shard_dir,seed=42)
    try:
        dataset=data['validation']['selection']
        indices=select_chunks(dataset,data['validation_components'],50)
        selected_recordings={dataset.chunks[i].recording_id for i in indices}
        indices=[i for i,c in enumerate(dataset.chunks) if c.recording_id in selected_recordings]
        ids=[dataset.chunks[i].chunk_id for i in indices]
        waves=waves_for(dataset,indices)
        rids=[dataset.chunks[i].recording_id for i in indices]
    finally:
        data['loader'].close()
    source=ROOT/'runs/r4/r4_u1_ast_ft_A_20261115T102637Z'
    expected={r['chunk_id']:r['tts_probability'] for r in pq.read_table(source/'predictions_chunks_selection.parquet').to_pylist()}
    adapter=ASTBundleAdapter.load(a.bundle,device='cuda',precision='fp16',batch_size=8)
    scores=adapter.score_chunks(waves)
    report=parity_check(ids,np.array([expected[i] for i in ids]),scores_fp16=scores)
    rule=bundle.spec.decision_rule
    flips=[]
    # Include every chunk of each sampled recording before calibration.
    comparisons=[]
    for rid in sorted(set(rids)):
        mask=np.array([r==rid for r in rids]); raw=float(np.mean(scores[mask])); ref=float(np.mean([expected[c] for c,r in zip(ids,rids) if r==rid]))
        q=float(mapping(meta['method'],meta['parameters'],[raw])[0]) if meta['accepted'] else raw
        qr=float(mapping(meta['method'],meta['parameters'],[ref])[0]) if meta['accepted'] else ref
        d=int(rule.kind=='threshold' and q>=rule.threshold); dr=int(rule.kind=='threshold' and qr>=rule.threshold)
        if d!=dr:flips.append(rid)
        comparisons.append({'recording_id':rid,'sample_chunk_mean':raw,'sample_calibrated_mean':q,'decision':d,'reference_decision':dr})
    report.update(bundle_sha256=digest(a.bundle/'bundle.json'),calibration_sha256=digest(bundle.components['calibration/ast_a.json']),
                  decision_flips=flips,comparisons=comparisons,scope='All chunks of recordings represented by 50 stratified selection chunks; mean then calibration then saved rule',
                  source_sha256=digest(__file__),reference_precision='cuda_fp16',approved=False)
    a.report.parent.mkdir(parents=True,exist_ok=True)
    write(a.report,report)
    if not report['passed'] or flips:raise SystemExit('Parity requires review')


if __name__=='__main__':main()
