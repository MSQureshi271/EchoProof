"""Supplementary fixed-bin reliability/threshold analysis; no fitting or selection."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
import uuid
from importlib.metadata import version

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from r3_n1_ast_preflight import digest, require, verify, RUN
from r3_n1_lr_operating import reliability
from echoproof.evaluation.calibrate import apply_temperature
from echoproof.evaluation.metrics import compute_metrics

CANDIDATE = 'runs/r3/r3_n1_ast_candidates/20261113T222724Z-ae8cab60'


def diagram(series):
    """Standalone SVG: mean predicted P(TTS) versus observed TTS frequency."""
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="650" height="570" viewBox="0 0 650 570">',
             '<rect width="650" height="570" fill="white"/>',
             '<g font-family="sans-serif" font-size="13">',
             '<text x="65" y="25">AST-B operating reliability: 10 fixed equal-width bins</text>',
             '<path d="M70 60V480H550" fill="none" stroke="black"/>',
             '<path d="M70 480L550 60" stroke="gray" stroke-dasharray="5 5"/>']
    for i in range(11):
        x, y = 70+48*i, 480-42*i
        parts += [f'<text x="{x-8}" y="500">{i/10:.1f}</text>', f'<text x="35" y="{y+4}">{i/10:.1f}</text>']
    for j, (name, bins) in enumerate(series.items()):
        color = ['#1565c0','#c62828'][j]
        points = [(70+480*b['mean_score'],480-420*b['positive_fraction'], b['count']) for b in bins if b['count']]
        coords = ' '.join(f'{x},{y}' for x,y,_ in points)
        parts += [f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2"/>',
                  f'<text x="380" y="{40+18*j}" fill="{color}">{name}</text>']
        for x,y,n in points:
            parts.append(f'<circle cx="{x}" cy="{y}" r="4" fill="{color}"><title>n={n}</title></circle>')
    parts += ['<text x="170" y="535">Mean predicted P(TTS) in each bin</text>',
              '<text transform="translate(18 370) rotate(-90)">Observed TTS fraction</text>',
              '<text x="70" y="560">Empty bins omitted; counts available in reliability.json. Development data.</text>', '</g></svg>']
    return '\n'.join(parts)


def run(root):
    import pyarrow.parquet as pq
    root = Path(root).resolve()
    candidate = root/CANDIDATE
    # Snapshot all candidate bytes, including model weights; never rewrite them.
    protected = {p: digest(p) for p in candidate.rglob('*') if p.is_file()}
    original = verify(root)
    evidence = json.loads((candidate/'evidence.json').read_text())
    for name, sha in evidence['outputs_sha256'].items():
        require(digest(candidate/name) == sha, f'Candidate artifact mismatch: {name}')
    source = root/RUN/'predictions_recordings_operating.parquet'
    rows = pq.read_table(source).to_pylist()
    labels, raw = [r['label'] for r in rows], [r['raw_score'] for r in rows]
    policy = json.loads((candidate/'prospective_policy.json').read_text())
    temperature = evidence['temperature']
    threshold = evidence['decision_rule']['threshold']
    scaled = apply_temperature(raw, temperature, epsilon=policy['epsilon'])
    saved = pq.read_table(candidate/'operating_predictions.parquet').to_pylist()
    require([r['recording_id'] for r in rows] == [r['recording_id'] for r in saved], 'Prediction order mismatch')
    require(scaled == [r['temperature_score'] for r in saved], 'Temperature reproduction mismatch')
    results, bins = {}, {}
    for name,scores in [('raw',raw),('temperature',scaled)]:
        bins[name], ece = reliability(labels,scores)
        results[name] = {'ece':ece, **{k:v for k,v in compute_metrics(labels,scores,.5).items() if k in ('brier_score','log_loss')}}
    base, chosen = compute_metrics(labels,raw,.5), compute_metrics(labels,raw,threshold)
    changed = [{ 'recording_id':r['recording_id'], 'group_id':r['group_id'], 'label':y, 'raw_score':s,
                 'old_decision':int(s>=.5), 'new_decision':int(s>=threshold)} for r,y,s in zip(rows,labels,raw) if (s>=.5)!=(s>=threshold)]
    report = {'status':'supplementary_analysis_not_acceptance', 'candidate':CANDIDATE,
              'temperature_fixed':temperature, 'threshold_fixed_full_precision':threshold,
              'ece_definition':'10 equal-width bins [i/10,(i+1)/10), last includes 1; sum(n_bin/N * abs(mean P(TTS)-TTS fraction)); empty bins contribute zero',
              'reliability':results, 'raw_threshold_0_5':base, 'raw_saved_threshold':chosen,
              'changed_decisions':len(changed), 'corrected':sum(r['new_decision']==r['label'] for r in changed),
              'worsened':sum(r['new_decision']!=r['label'] for r in changed),
              'no_refit_or_threshold_search':True, 'candidate_acceptance_unchanged':True,
              'limitations':['ECE depends on binning; no post-hoc bin search', 'Operating threshold performance is selection-biased development evidence', 'No waveform precision parity established'],
              'environment':{'python':platform.python_version(), **{m:version(m) for m in ('numpy','pyarrow')}},
              'input_sha256':{str(p.relative_to(root)):s for p,s in protected.items()},
              'source_sha256':{str(p.relative_to(root)):digest(p) for p in [Path(__file__),root/'scripts/r3_n1_lr_operating.py', root/'scripts/r3_n1_ast_preflight.py',root/'src/echoproof/evaluation/calibrate.py',root/'src/echoproof/evaluation/metrics.py']}}
    require(all(digest(p)==s for p,s in protected.items()), 'Candidate changed')
    require(verify(root)['hashes']==original['hashes'], 'Original inputs changed')
    report['original_inputs_sha256'] = original['hashes']
    out = root/'runs/r3/r3_n1_ast_supplementary'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8])
    out.mkdir(parents=True,exist_ok=False)
    for name,value in [('reliability.json',bins),('changed_decisions.json',changed)]:
        (out/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    (out/'reliability.svg').write_text(diagram(bins),encoding='utf-8')
    report['output_sha256']={p.name:digest(p) for p in out.iterdir()}
    (out/'evidence.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(out)
    print(json.dumps({k:report[k] for k in ['reliability','raw_threshold_0_5','raw_saved_threshold','changed_decisions','corrected','worsened']},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,required=True)
    run(parser.parse_args().project_root)
