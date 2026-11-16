"""Descriptive R5-M1 analysis of saved sealed predictions and audio measurements."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(root, output):
    root, output = Path(root), Path(output)
    sealed = root / 'reports/final/results/r5_n1_clean_v1'
    quality = root / 'reports/final/results/r5_m1_error_taxonomy_v1/quality_v1'
    report = json.loads((quality / 'measurement_report.json').read_text())
    integrity = json.loads((sealed / 'integrity_report.json').read_text())
    if digest(quality / 'recording_quality.parquet') != report['output_sha256']:
        raise ValueError('Quality table hash mismatch')
    if digest(sealed / 'integrity_report.json') != report['integrity_sha256']:
        raise ValueError('Sealed integrity evidence mismatch')
    comparison_path = sealed / 'analysis_clean_v1/comparison.csv'
    hashes = json.loads((sealed / 'analysis_clean_v1/output_hashes.json').read_text())
    if digest(comparison_path) != hashes['comparison.csv']:
        raise ValueError('Comparison table hash mismatch')
    comparison = pd.read_csv(comparison_path)
    measurements = pd.read_parquet(quality / 'recording_quality.parquet')
    if measurements.recording_id.duplicated().any() or not measurements.status.eq('success').all():
        raise ValueError('Quality measurements incomplete or duplicated')
    rows = []
    for protocol in ('A', 'B'):
        for model in ('LR', 'BiGRU'):
            entry = integrity['bundles'][f'{protocol}:{model}']
            path = sealed / entry['predictions_file']
            if not entry['integrity']['passed'] or digest(path) != entry['predictions_sha256'] or digest(path) != report['prediction_sha256'][path.name]:
                raise ValueError(f'Unverified predictions {path}')
            panel = 'A_known' if protocol == 'A' else 'B_known'
            threshold = float(comparison.query('panel == @panel and model == @model').iloc[0].threshold)
            df = pd.read_parquet(path).merge(measurements.drop(columns=['status', 'error']), on='recording_id', how='left', validate='one_to_one')
            if df.duration_s.isna().any() or not df.status.eq('success').all() or not np.allclose(df.duration_s, df.decoded_duration_seconds, atol=1/16000, rtol=0):
                raise ValueError(f'Incomplete or inconsistent measurements for {path}')
            decision = df.score >= threshold
            if not (decision == df.assessment.eq('synthetic')).all():
                raise ValueError(f'Decisions do not match locked threshold for {path}')
            df['protocol'], df['model'], df['threshold'] = protocol, model, threshold
            df['error_type'] = np.select([(df.label == 0) & decision, (df.label == 1) & ~decision, (df.label == 0) & ~decision], ['false_positive', 'false_negative', 'true_negative'], default='true_positive')
            df['duration_band'] = pd.cut(df.duration_s, [0, 4, 8, 12, np.inf], right=False, labels=['<4s','4–<8s','8–<12s','≥12s']).astype(str)
            df['silence_band'] = pd.cut(df.silence_fraction, [-np.inf, .1, .3, .5, np.inf], right=False, labels=['<10%','10–<30%','30–<50%','≥50%']).astype(str)
            df['level_band'] = pd.cut(df.rms_dbfs, [-np.inf, -30, -20, -10, np.inf], right=False, labels=['<-30 dBFS','-30–<-20 dBFS','-20–<-10 dBFS','≥-10 dBFS']).astype(str)
            df['clipping_band'] = np.where(df.clip_fraction > 0, 'clipped samples present', 'none detected')
            rows.append(df)
    all_rows = pd.concat(rows, ignore_index=True)
    # The B genuine set occurs once here; family component keeps denominators separate.
    all_rows['is_error'] = all_rows.error_type.isin(['false_positive','false_negative'])
    tables = {}
    for variable in ('component', 'duration_band', 'silence_band', 'level_band', 'clipping_band', 'group_id'):
        tables[variable] = all_rows.groupby(['protocol','model','label',variable], observed=True, dropna=False).agg(recordings=('recording_id','size'), errors=('is_error','sum'), speakers=('group_id','nunique')).reset_index()
        tables[variable]['error_rate'] = tables[variable].errors / tables[variable].recordings
    for variable in ('duration_band', 'silence_band', 'level_band', 'clipping_band'):
        key = f'component_{variable}'
        tables[key] = all_rows.groupby(['protocol','model','component',variable], observed=True, dropna=False).agg(recordings=('recording_id','size'), errors=('is_error','sum'), speakers=('group_id','nunique')).reset_index()
        tables[key]['error_rate'] = tables[key].errors / tables[key].recordings
    pairs = all_rows[all_rows.model.eq('LR')].merge(all_rows[all_rows.model.eq('BiGRU')], on=['protocol','recording_id'], suffixes=('_lr','_bigru'), validate='one_to_one')
    if not (pairs.label_lr == pairs.label_bigru).all():
        raise ValueError('Paired labels disagree')
    pairs['pattern'] = np.select([pairs.is_error_lr & pairs.is_error_bigru, pairs.is_error_lr & ~pairs.is_error_bigru, ~pairs.is_error_lr & pairs.is_error_bigru], ['both wrong','LR only wrong','BiGRU only wrong'], default='both correct')
    pair_counts = pairs.groupby(['protocol','component_lr','pattern'], observed=True).size().rename('recordings').reset_index()
    # Stable examples are candidates for human listening, not endorsed demo clips.
    candidates = pairs.sort_values('recording_id').groupby(['protocol','component_lr','pattern'], observed=True).head(3)
    candidate_cols = ['protocol','recording_id','group_id_lr','component_lr','label_lr','duration_s_lr','pattern','score_lr','threshold_lr','assessment_lr','score_type_lr','score_bigru','threshold_bigru','assessment_bigru','score_type_bigru']
    if output.exists():
        raise FileExistsError(f'Output exists: {output}')
    output.mkdir(parents=True)
    all_rows.to_parquet(output/'recording_errors.parquet', index=False)
    for name, table in tables.items(): table.to_csv(output/f'by_{name}.csv', index=False)
    pair_counts.to_csv(output/'paired_errors.csv', index=False)
    candidates[candidate_cols].to_csv(output/'demo_candidates_unreviewed.csv', index=False)
    provenance = {'scope':'descriptive analysis of saved predictions; no inference or tuning', 'prediction_sha256':report['prediction_sha256'], 'quality_sha256':report['output_sha256'], 'comparison_sha256':hashes['comparison.csv'], 'script_sha256':digest(Path(__file__)), 'rows':len(all_rows), 'duration_bins':'[0,4), [4,8), [8,12), [12,infinity) seconds', 'silence_bins':'[0,.1), [.1,.3), [.3,.5), [.5,1]', 'level_bins':'[-infinity,-30), [-30,-20), [-20,-10), [-10,infinity) dBFS', 'bins_note':'chosen for descriptive analysis after sealed results; not preregistered', 'demo_note':'candidates selected deterministically by recording ID; listening and suitability review pending'}
    (output/'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n', encoding='utf-8')
    print(f'Wrote {len(all_rows)} prediction rows to {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.root, args.output)
