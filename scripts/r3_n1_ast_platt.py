"""Build an isolated team-authorized AST Platt candidate; never updates locks."""
import argparse
import importlib.util
import json
import platform
import sys
import uuid
from dataclasses import replace
from datetime import datetime,timezone
from importlib.metadata import version
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from r3_n1_ast_preflight import verify,digest,require,RUN,CHECKPOINT
from r3_n1_ast_calibrate import canonical_check
from r3_n1_ast_supplement import diagram
from echoproof.bundle import load_bundle,write_bundle
from echoproof.models.r3_n1_ast_platt_bundle import platt_score,inspect_ast_platt_candidate
from echoproof.evaluation.metrics import compute_metrics
from echoproof.evaluation.threshold import select_threshold
from r3_n1_lr_operating import reliability


def execute():
    policy_path=ROOT/'configs/r3_n1_ast_platt_candidate.json'
    policy=json.loads(policy_path.read_text())
    require(policy['epsilon']==1e-7 and policy['checkpoint_sha256']==CHECKPOINT,'Unexpected candidate policy')
    pre=verify(ROOT);canonical_check(ROOT,ROOT/RUN)
    protected={ROOT/p:s for p,s in pre['hashes'].items()}
    for p in [ROOT/'locks/B_lock.json',ROOT/'bundles/r3_u2_ast_partial_B/bundle.json',ROOT/RUN/'selection.json',ROOT/RUN/'evidence.json']:
        protected[p]=digest(p)
    old=ROOT/'runs/r3/r3_n1_ast_candidates/20261113T222724Z-ae8cab60'
    protected.update({p:digest(p) for p in old.rglob('*') if p.is_file()})
    exp=ROOT/'experiments/r3_n1/ast_alternative_calibration'
    reference=exp/'runs/20261114T112414Z-7acaafec'
    record=json.loads((reference/'prospective.json').read_text())
    for name,sha in record['sources'].items():require(digest(ROOT/name)==sha,f'Exploratory source mismatch: {name}')
    for name,sha in json.loads((reference/'manifest.json').read_text()).items():
        require(digest(reference/name)==sha,f'Exploratory artifact mismatch: {name}')
        protected[reference/name]=sha
    spec=importlib.util.spec_from_file_location('ast_platt_verified_exploration',exp/'run.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    source_paths=[Path(__file__),policy_path,ROOT/'src/echoproof/models/r3_n1_ast_platt_bundle.py',ROOT/'tests/test_r3_n1_ast_platt.py',
                  ROOT/'scripts/r3_n1_ast_preflight.py',ROOT/'scripts/r3_n1_ast_calibrate.py',ROOT/'scripts/r3_n1_lr_operating.py',ROOT/'scripts/r3_n1_ast_supplement.py',
                  *[ROOT/f'src/echoproof/{name}.py' for name in ['evaluation/calibrate','evaluation/metrics','evaluation/threshold','bundle','inference','models/ast_adapter','models/ast_detector']],exp/'run.py']
    out=ROOT/'runs/r3/r3_n1_ast_platt_candidates'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8])
    out.mkdir(parents=True,exist_ok=False)
    def write(name,value):
        (out/name).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    write('prospective.json',{'policy':policy,'sources':{p.relative_to(ROOT).as_posix():digest(p) for p in source_paths},
        'inputs':{p.relative_to(ROOT).as_posix():s for p,s in protected.items()},'environment':{'python':platform.python_version(),**{m:version(m) for m in ['numpy','scipy','scikit-learn','pyarrow']}}})
    rows={role:sorted(pq.read_table(ROOT/RUN/f'predictions_recordings_{role}.parquet').to_pylist(),key=lambda r:r['recording_id']) for role in ['calibration','operating']}
    cal=rows['calibration'];op=rows['operating']
    params=module.fit('platt',np.array([r['label'] for r in cal]),np.array([r['raw_score'] for r in cal]))
    expected=next(r['params'] for r in json.loads((reference/'operating/results.json').read_text())['folds'] if r['method']=='platt')
    require(all(abs(params[k]-expected[k])<1e-12 for k in ['a','b']),'Fit does not reproduce verified exploration')
    y=[r['label'] for r in op];raw=[r['raw_score'] for r in op]
    scaled=[platt_score(p,params['a'],params['b']) for p in raw]
    np.testing.assert_allclose(scaled,module.apply('platt',params,np.array(raw)),atol=1e-15,rtol=0)
    rule,f1,fpr=select_threshold(y,scaled)
    fit_record={'method':'platt_on_recording_log_odds',**params,'epsilon':1e-7,'fit_role':'calibration','recordings':len(cal),'speakers':len({r['group_id'] for r in cal}),
                'calibration_predictions_sha256':digest(ROOT/RUN/'predictions_recordings_calibration.parquet')}
    write('fit.json',fit_record)
    meta={'schema':'ast_platt_candidate_v1','method':'platt_on_recording_log_odds','a':params['a'],'b':params['b'],'epsilon':1e-7,
          'checkpoint_sha256':CHECKPOINT,'policy_sha256':digest(policy_path),'fit_sha256':digest(out/'fit.json'),
          'decision_rule':rule.to_dict(),'reference_precision':'fp16','officially_promoted':False}
    original=load_bundle(ROOT/'bundles/r3_u2_ast_partial_B')
    components=dict(original.components);components['calibration/platt.json']=json.dumps(meta,allow_nan=False).encode()
    write_bundle(out/'bundle',replace(original.spec,bundle_id='bundle:ast-platt-candidate-'+out.name,decision_rule=rule),components,
                 extra={'stage':'candidate_pending_integration_review','team_authorization':'reported_by_user','officially_promoted':False})
    loaded,m=inspect_ast_platt_candidate(out/'bundle')
    got=[platt_score(p,m['a'],m['b'],m['epsilon']) for p in raw]
    require(got==scaled,'Reload probability mismatch')
    require([loaded.spec.decision_rule.decide(p) for p in got]==[rule.decide(p) for p in scaled],'Reload decision mismatch')
    bins={name:reliability(y,p)[0] for name,p in [('raw',raw),('platt',scaled)]}
    write('reliability.json',{name:{'bins':bins[name],'ece':reliability(y,p)[1]} for name,p in [('raw',raw),('platt',scaled)]})
    (out/'reliability.svg').write_text(diagram(bins),encoding='utf-8')
    pq.write_table(pa.Table.from_pylist([{**r,'platt_score':p,'decision':rule.decide(p)} for r,p in zip(op,scaled)]),out/'operating_predictions.parquet')
    report={'status':'candidate_built_not_promoted','fit':fit_record,'decision_rule':rule.to_dict(),
            'raw_at_0_5':compute_metrics(y,raw,.5),'platt_at_0_5':compute_metrics(y,scaled,.5),
            'platt_at_selected_threshold':compute_metrics(y,scaled,rule.threshold) if rule.threshold is not None else {'kind':'all_negative','macro_f1':f1,'fpr':fpr},
            'probability_and_decision_reload_parity':True,'canonical_membership_verified':True,
            'waveform_inference':'not_executed: local torch and transformers unavailable; CUDA/fp16 export-reference verification pending',
            'historical_significance_gates':'not passed; unchanged; method authorized separately as reported by user',
            'promotion_requirements':['Actual AST waveform-to-decision parity at selected threshold','Documented recipient/reviewer integration approval','Separate explicit promotion; no lock updates performed']}
    require(all(digest(p)==s for p,s in protected.items()),'Protected artifacts changed')
    report['originals_unchanged']=True
    write('evidence.json',report)
    write('manifest.json',{p.relative_to(out).as_posix():digest(p) for p in out.rglob('*') if p.is_file()})
    print(out);print(json.dumps(report,indent=2))


if __name__=='__main__':execute()
