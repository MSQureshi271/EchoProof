"""Export raw validation predictions only after batch selection is fixed."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import EXPERIMENT, verify_contract
from prepare_batch import sha


def export(contract_path, shard_dir):
    import torch
    import pyarrow as pa
    import pyarrow.parquet as pq
    from echoproof.models.bigru_waveform import WaveformBiGRU
    from echoproof.training.r3_m1_data import build_data
    from echoproof.training.r3_m1_export import aggregate_role, score_dataset, write_json
    from echoproof.training.r3_m1_selection import selection_metrics
    contract_path = Path(contract_path)
    contract = json.loads(contract_path.read_text())
    verify_contract(contract, contract_path)
    if not torch.cuda.is_available():
        raise ValueError('CUDA required for checkpoint export parity')
    selected_path = EXPERIMENT/'reports/final_selection.json'
    selected = json.loads(selected_path.read_text())
    comparison = json.loads((EXPERIMENT/'reports/batch01_comparison.json').read_text())
    if comparison['selected'] != selected or comparison['contract_sha256'] != sha(contract_path):
        raise ValueError('decision and comparison disagree')
    checkpoint_path = (ROOT / selected['checkpoint']).resolve()
    if not checkpoint_path.is_relative_to(ROOT) or sha(checkpoint_path) != selected['checkpoint_sha256']:
        raise ValueError('selected checkpoint hash mismatch')
    data = build_data(ROOT, shard_dir, seed=42)
    out = EXPERIMENT/'runs/batch01/selected'
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        if data['input_sha256'] != contract['inputs']:
            raise ValueError('inputs changed')
        model = WaveformBiGRU(data['mel_stats']['mean'], data['mel_stats']['std']).cuda()
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        model.load_state_dict(checkpoint['model'], strict=True)
        products, counts = {}, {}
        for role in ('selection', 'calibration', 'operating'):
            if time.monotonic()-started >= 30*60:
                raise RuntimeError('export time budget reached')
            rows = score_dataset(model, data['validation'][role], 'cuda')
            recordings = aggregate_role(data['validation'][role], rows)
            if role == 'selection':
                metric = selection_metrics([r[1] for r in recordings], [r[2] for r in recordings], .5)['macro_f1']
                if abs(metric-selected['macro_f1']) > 1e-12:
                    raise ValueError('selected checkpoint reload metric differs')
            common = lambda rid: {'recording_id': rid, 'group_id': data['validation_groups'][rid],
                                  'component': data['validation_components'][rid], 'validation_role': role}
            tables = {'chunks': [{**common(rid), 'chunk_id': cid, 'label': y, 'tts_probability': s} for cid,rid,y,s in rows],
                      'recordings': [{**common(rid), 'label': y, 'raw_score': s, 'chunk_count': n} for rid,y,s,n in recordings]}
            for kind, table in tables.items():
                path = out / f'predictions_{kind}_{role}.parquet'
                tmp = path.with_suffix('.parquet.tmp')
                pq.write_table(pa.Table.from_pylist(table), tmp)
                os.replace(tmp, path)
                products[path.name] = sha(path)
            counts[role] = {'chunks': len(rows), 'recordings': len(recordings)}
        evidence = {'status': 'handoff_predictions_complete', 'selected': selected, 'contract_sha256': sha(contract_path),
                    'selection_decision_sha256': sha(selected_path), 'outputs': products, 'counts': counts,
                    'export_seconds': time.monotonic()-started, 'review_status': 'pending',
                    'score_type': 'uncalibrated raw scores; R3-N1 still required'}
        write_json(EXPERIMENT/'reports/final_handoff.json', evidence)
        return evidence
    finally:
        data['loader'].close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--contract', type=Path, required=True)
    p.add_argument('--shard-dir', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(export(a.contract, a.shard_dir), indent=2))
