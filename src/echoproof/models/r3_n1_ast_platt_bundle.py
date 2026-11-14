"""Explicit AST Platt candidate interface; no automatic app or lock promotion."""
import json
import math
from numbers import Real
from scipy.special import expit
from echoproof.bundle import load_bundle, BundleError
from echoproof.data.chunks import PREPROCESSING_HASH


def platt_score(score, a, b, epsilon=1e-7):
    values=(score,a,b,epsilon)
    if any(isinstance(x,bool) or not isinstance(x,Real) or not math.isfinite(x) for x in values):
        raise ValueError('Finite numeric Platt inputs required')
    if not 0<=score<=1 or a<=0 or epsilon!=1e-7:
        raise ValueError('Invalid Platt score, slope or clipping convention')
    p=min(max(float(score),epsilon),1-epsilon)
    return float(expit(a*(math.log(p)-math.log1p(-p))+b))


def inspect_ast_platt_candidate(path):
    bundle=load_bundle(path)
    if bundle.spec.model_version!='ast_partial:B' or bundle.spec.preprocessing_hash!=PREPROCESSING_HASH:
        raise BundleError('Wrong AST model/preprocessing')
    required={'model/ast_detector.json','model/config.json','model/preprocessor_config.json','model/model.safetensors','calibration/platt.json'}
    if set(bundle.components)!=required:
        raise BundleError('Unexpected candidate component set')
    meta=json.loads(bundle.components['calibration/platt.json'].read_text())
    model=json.loads(bundle.components['model/ast_detector.json'].read_text())
    if (meta.get('schema')!='ast_platt_candidate_v1' or meta.get('method')!='platt_on_recording_log_odds'
        or meta.get('checkpoint_sha256')!=model.get('source_checkpoint_sha256')
        or meta.get('decision_rule')!=bundle.spec.decision_rule.to_dict()
        or meta.get('reference_precision')!='fp16' or meta.get('officially_promoted') is not False):
        raise BundleError('Inconsistent AST Platt provenance or rule')
    platt_score(.5,meta['a'],meta['b'],meta['epsilon'])
    return bundle,meta


def load_ast_platt_candidate(path, *, device='cuda', precision='fp16'):
    from echoproof.models.ast_adapter import ASTBundleAdapter
    from echoproof.inference import LoadedBundle
    bundle,meta=inspect_ast_platt_candidate(path)
    if precision!='fp16' or not str(device).startswith('cuda'):
        raise ValueError('Only export-reference CUDA/fp16 is enabled; CPU/fp32 decision parity remains unverified')
    model=ASTBundleAdapter.load(path,device=device,precision=precision)
    # LoadedBundle/analyze performs mean(chunk scores), then this transform, then decision.
    return LoadedBundle(spec=bundle.spec,scorer=model.score_chunks,
        calibrate=lambda raw:platt_score(raw,meta['a'],meta['b'],meta['epsilon']),
        score_type='calibrated_probability')
