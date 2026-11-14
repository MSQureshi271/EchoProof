"""Fixed-design exploratory AST calibration; writes only unique experiment runs."""
from pathlib import Path
import sys, json, csv, uuid, platform, warnings
from datetime import datetime, timezone
from importlib.metadata import version
import numpy as np
import pyarrow.parquet as pq
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

EXP = Path(__file__).resolve().parent
ROOT = EXP.parents[2]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
from echoproof.evaluation.calibrate import fit_temperature, apply_temperature
from echoproof.evaluation.threshold import select_threshold
from echoproof.evaluation.metrics import compute_metrics
from r3_n1_ast_preflight import digest, require, verify, RUN
from r3_n1_ast_calibrate import canonical_check

EPS = 1e-7
METHODS = ('raw','temperature','platt','beta','isotonic')
SAVED = ROOT/'runs/r3/r3_n1_ast_candidates/20261113T222724Z-ae8cab60'


def features(p, method):
    p = np.clip(p, EPS, 1-EPS)
    if method == 'platt':
        return np.column_stack((np.log(p)-np.log1p(-p), np.ones(len(p))))
    return np.column_stack((np.log(p), -np.log1p(-p), np.ones(len(p))))


def fit(method, y, p):
    require(set(y)=={0,1}, 'Both fit classes required')
    if method=='raw': return {}
    if method=='temperature':
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            t = fit_temperature(y.tolist(),p.tolist(),epsilon=EPS,bounds=(.01,50.))
        return {'temperature':t, 'warnings':[str(w.message) for w in captured], 'at_boundary':t<=.01001 or t>=49.99999}
    if method=='isotonic':
        iso=IsotonicRegression(increasing=True,out_of_bounds='clip').fit(p,y)
        return {'x':iso.X_thresholds_.tolist(),'y':iso.y_thresholds_.tolist(),
                'knots':len(iso.X_thresholds_), 'levels':len(np.unique(iso.y_thresholds_))}
    x=features(p,method)
    initial=[1.,0.] if method=='platt' else [1.,1.,0.]
    bounds=[(1e-8,None),(None,None)] if method=='platt' else [(0,None),(0,None),(None,None)]
    def objective(v):
        z=x@v
        return float(np.mean(np.logaddexp(0,z)-y*z)), x.T@(expit(z)-y)/len(y)
    result=minimize(objective,initial,jac=True,method='L-BFGS-B',bounds=bounds,
                    options={'maxiter':2000,'ftol':1e-12,'gtol':1e-8})
    require(result.success and np.isfinite(result.x).all(), f'{method} optimizer failed: {result.message}')
    keys=['a','b'] if method=='platt' else ['a','b','c']
    return {**dict(zip(keys,map(float,result.x))), 'iterations':int(result.nit),
            'at_boundary':any(lo is not None and abs(v-lo)<1e-7 for v,(lo,_) in zip(result.x,bounds)),
            'objective':float(result.fun),'optimizer_message':str(result.message)}


def apply(method, params, p):
    if method=='raw': return p.copy()
    if method=='temperature': return np.asarray(apply_temperature(p.tolist(),params['temperature'],epsilon=EPS))
    if method=='isotonic': return np.clip(np.interp(p,params['x'],params['y']),EPS,1-EPS)
    keys=['a','b'] if method=='platt' else ['a','b','c']
    return expit(features(p,method)@np.array([params[k] for k in keys]))


def reliability(y,p,equal_count=False):
    if equal_count:
        indices=np.array_split(np.argsort(p,kind='stable'),10)
    else:
        which=np.minimum((p*10).astype(int),9)
        indices=[np.flatnonzero(which==i) for i in range(10)]
    rows=[]
    for i,idx in enumerate(indices):
        rows.append({'bin':i,'count':len(idx),'mean_score':float(p[idx].mean()) if len(idx) else None,
                     'tts_fraction':float(y[idx].mean()) if len(idx) else None,
                     'score_min':float(p[idx].min()) if len(idx) else None,'score_max':float(p[idx].max()) if len(idx) else None})
    return float(sum(r['count']*abs(r['mean_score']-r['tts_fraction']) for r in rows if r['count'])/len(y)),rows


def loss_arrays(y,p):
    c=np.clip(p,EPS,1-EPS)
    return np.column_stack((-(y*np.log(c)+(1-y)*np.log1p(-c)),(p-y)**2))


def scores(y,p):
    require(np.isfinite(p).all() and ((0<=p)&(p<=1)).all(),'Invalid prediction')
    nll,brier=loss_arrays(y,p).mean(axis=0)
    return {'nll':float(nll),'brier':float(brier),'ece_width':reliability(y,p)[0],
            'ece_count':reliability(y,p,True)[0],'auroc':float(roc_auc_score(y,p))}


def folds(groups,k=None,seed=42):
    speakers=np.unique(groups)
    if k is None: return [np.array([g]) for g in speakers]
    return np.array_split(np.random.default_rng(seed).permutation(speakers),k)


def summary_params(records):
    result={}
    for method in METHODS:
        records_m=[r['params'] for r in records if r['method']==method]
        keys=sorted({k for p in records_m for k,v in p.items() if type(v) in (float,int)})
        result[method]={k:{'min':float(np.min(v:=[p[k] for p in records_m if k in p])),
                           'max':float(np.max(v)),'std':float(np.std(v))} for k in keys}
        result[method]['boundary_fits']=sum(p.get('at_boundary',False) for p in records_m)
    return result


def bootstrap(y,raw,pred,groups):
    unique=np.unique(groups); rng=np.random.default_rng(42)
    group_idx={g:np.flatnonzero(groups==g) for g in unique}
    estimates={m:[] for m in METHODS if m!='raw'}
    base=loss_arrays(y,raw)
    losses={m:loss_arrays(y,pred[m])-base for m in estimates}
    for _ in range(1000):
        idx=np.concatenate([group_idx[g] for g in rng.choice(unique,len(unique),replace=True)])
        # Preserve original recording-ID order for deterministic handling of ties.
        idx.sort(); yy=y[idx]; rr=raw[idx]
        raw_ece=np.array([reliability(yy,rr)[0],reliability(yy,rr,True)[0]])
        for m in estimates:
            pp=pred[m][idx]
            estimates[m].append(np.r_[losses[m][idx].mean(axis=0),
                np.array([reliability(yy,pp)[0],reliability(yy,pp,True)[0]])-raw_ece])
    return {m:{key:{'delta':scores(y,pred[m])[key]-scores(y,raw)[key],
                          'ci95':np.quantile(np.asarray(values)[:,j],[.025,.975]).tolist()}
               for j,key in enumerate(('nll','brier','ece_width','ece_count'))} for m,values in estimates.items()}


def influence(y,raw,pred,groups):
    result={}
    for m in METHODS[1:]:
        d=loss_arrays(y,pred[m])-loss_arrays(y,raw)
        result[m]=[{'speaker':g,'n':int(sum(groups==g)),
                    'mean_delta_nll':float(d[groups==g,0].mean()),'mean_delta_brier':float(d[groups==g,1].mean()),
                    'overall_delta_without_speaker':d[groups!=g].mean(axis=0).tolist()} for g in np.unique(groups)]
    return result


def write_json(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def evaluate(out,name,rows,pred,records,fold_ids):
    dest=out/name; dest.mkdir()
    y=np.array([r['label'] for r in rows]); groups=np.array([r['group_id'] for r in rows])
    result={'metrics':{m:scores(y,pred[m]) for m in METHODS},'folds':records,
            'parameter_summary':summary_params(records),'bootstrap':bootstrap(y,pred['raw'],pred,groups),
            'speaker_influence':influence(y,pred['raw'],pred,groups),
            'reliability':{m:{'equal_width':reliability(y,pred[m])[1], 'equal_count':reliability(y,pred[m],True)[1]} for m in METHODS}}
    with (dest/'predictions.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(['recording_id','speaker','original_role','label','fold',*METHODS])
        for i,r in enumerate(rows):w.writerow([r['recording_id'],r['group_id'],r['validation_role'],r['label'],int(fold_ids[i]),*[float(pred[m][i]) for m in METHODS]])
    write_json(dest/'results.json',result)
    print(name, json.dumps(result['metrics']),flush=True)
    return result


def cv(out,name,rows,k=None,seed=42):
    y=np.array([r['label'] for r in rows]); p=np.array([r['raw_score'] for r in rows]); g=np.array([r['group_id'] for r in rows])
    pred={m:np.full(len(y),np.nan) for m in METHODS}; fold_ids=np.full(len(y),-1); records=[]
    for f,held in enumerate(folds(g,k,seed)):
        test=np.isin(g,held); train=~test
        require(set(y[train])==set(y[test])=={0,1},f'{name}/{f}: missing class; abort, do not drop fold')
        require(not set(g[train])&set(g[test]),'Speaker overlap')
        require((fold_ids[test]==-1).all(),'Duplicate OOF allocation'); fold_ids[test]=f
        for m in METHODS:
            params=fit(m,y[train],p[train]); pred[m][test]=apply(m,params,p[test])
            records.append({'fold':f,'method':m,'held_speakers':held.tolist(),'fit_speakers':sorted(set(g[train])),
                            'fit_counts':np.bincount(y[train],minlength=2).tolist(),'held_counts':np.bincount(y[test],minlength=2).tolist(),
                            'params':params,'metrics':scores(y[test],pred[m][test]),
                            'raw_fold_auroc':float(roc_auc_score(y[test],p[test]))})
    require((fold_ids>=0).all(),'Missing OOF rows')
    return evaluate(out,name,rows,pred,records,fold_ids)


def main():
    pre=verify(ROOT); canonical_check(ROOT,ROOT/RUN)
    source_paths=[EXP/'PLAN.md',Path(__file__),EXP/'test_experiment.py',
        ROOT/'scripts/r3_n1_ast_preflight.py',ROOT/'scripts/r3_n1_ast_calibrate.py',
        *[ROOT/f'src/echoproof/evaluation/{x}.py' for x in ('calibrate','metrics','threshold')]]
    protected={ROOT/p:sha for p,sha in pre['hashes'].items()}
    protected.update({p:digest(p) for p in SAVED.rglob('*') if p.is_file()})
    for p in [ROOT/'locks/B_lock.json',ROOT/'bundles/r3_u2_ast_partial_B/bundle.json',ROOT/RUN/'selection.json',ROOT/RUN/'evidence.json']:
        protected[p]=digest(p)
    candidate=json.loads((SAVED/'evidence.json').read_text())
    for name,sha in candidate['outputs_sha256'].items():require(digest(SAVED/name)==sha,'Candidate changed')
    rows={role:sorted(pq.read_table(ROOT/RUN/f'predictions_recordings_{role}.parquet').to_pylist(),key=lambda r:r['recording_id']) for role in ('calibration','operating')}
    for role,n,ng in [('calibration',960,12),('operating',1040,13)]:
        require(len(rows[role])==n and len({r['group_id'] for r in rows[role]})==ng,'Unexpected coverage')
    require(not {r['group_id'] for r in rows['calibration']}&{r['group_id'] for r in rows['operating']},'Roles overlap')
    out=EXP/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]); out.mkdir(parents=True)
    write_json(out/'prospective.json',{'plan':(EXP/'PLAN.md').read_text(),'sources':{str(p.relative_to(ROOT)):digest(p) for p in source_paths},
        'inputs':{str(p.relative_to(ROOT)):s for p,s in protected.items()},'status':'exploratory_not_accepted',
        'environment':{'python':platform.python_version(),**{m:version(m) for m in ('numpy','scipy','scikit-learn','pyarrow')}}})
    results={}
    results['calibration_loso']=cv(out,'calibration_loso',rows['calibration'])
    for seed in (42,43,44):results[f'calibration_4fold_{seed}']=cv(out,f'calibration_4fold_{seed}',rows['calibration'],4,seed)
    # Fit on calibration only, evaluate operating without any outcome-guided choices.
    cal=rows['calibration']; op=rows['operating']; cy=np.array([r['label'] for r in cal]); cp=np.array([r['raw_score'] for r in cal]); oy=np.array([r['label'] for r in op]); raw=np.array([r['raw_score'] for r in op])
    pred={}; records=[]; classification={}
    for m in METHODS:
        params=fit(m,cy,cp); pred[m]=apply(m,params,raw)
        records.append({'method':m,'params':params})
        rule,f1,fpr=select_threshold(oy.tolist(),pred[m].tolist())
        fixed=compute_metrics(oy.tolist(),pred[m].tolist(),candidate['decision_rule']['threshold'])
        decisions=np.array([rule.decide(float(x)) for x in pred[m]])
        classification[m]={'fixed_saved_threshold':fixed,'exploratory_optimized_rule':rule.to_dict(),
                           'optimized_macro_f1':f1,'optimized_fpr':fpr,'optimized_accuracy':float((decisions==oy).mean()),
                           'optimized_fnr':float((decisions[oy==1]==0).mean()),'distinct_scores':len(np.unique(pred[m])),
                           'nondecreasing_on_sorted_raw':bool((np.diff(pred[m][np.argsort(raw)])>=0).all())}
    results['operating']=evaluate(out,'operating',op,pred,records,np.zeros(len(op),dtype=int))
    write_json(out/'operating/classification.json',classification)
    pooled=sorted(cal+op,key=lambda r:r['recording_id'])
    for seed in (42,43,44):results[f'pooled_5fold_{seed}']=cv(out,f'pooled_5fold_{seed}',pooled,5,seed)
    gates={}
    for m in METHODS[1:]:
        r=results['operating']['metrics']; pooled_results=[results[f'pooled_5fold_{s}'] for s in (42,43,44)]
        gates[m]={'pooled_nll_brier_lower_all_seeds':all(v['metrics'][m]['nll']<v['metrics']['raw']['nll'] and v['metrics'][m]['brier']<v['metrics']['raw']['brier'] for v in pooled_results),
                  'pooled_nll_ci_below_zero_all_seeds':all(v['bootstrap'][m]['nll']['ci95'][1]<0 for v in pooled_results),
                  'operating_both_lower':r[m]['nll']<r['raw']['nll'] and r[m]['brier']<r['raw']['brier'],
                  'operating_candidate_rule':r[m]['nll']<r['raw']['nll'] and r[m]['brier']<=r['raw']['brier'],
                  'parameter_stability':'review_required_no_predeclared_tolerance',
                  'verdict':'no reliable improvement established under all required gates; not accepted'}
    require(all(digest(p)==s for p,s in protected.items()),'Protected artifact changed')
    write_json(out/'summary.json',{'status':'exploratory_complete_not_accepted','gates':gates,
        'metrics':{name:r['metrics'] for name,r in results.items()},'originals_unchanged':True,
        'limitations':['Operating and pooled results are post-hoc, not independent evaluation','Bootstrap fixes fitted models/OOF predictions and omits fitting uncertainty','CV seeds share data; no multiplicity correction','Parameter-stability gate needs review']})
    write_json(out/'manifest.json',{str(p.relative_to(out)):digest(p) for p in out.rglob('*') if p.is_file()})
    print('OUTPUT',out,flush=True)


if __name__=='__main__':main()
