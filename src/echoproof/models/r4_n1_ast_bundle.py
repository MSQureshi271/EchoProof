"""Explicit A-only AST calibration candidate loader; no production routing change."""
import json
from pathlib import Path
from echoproof.bundle import load_bundle, write_bundle
from echoproof.contracts import BundleSpecification, DecisionRule
from echoproof.evaluation.r4_n1_ast import digest, mapping

COMPONENT = 'calibration/ast_a.json'


def inspect(path):
    bundle = load_bundle(path)
    if bundle.spec.model_version != 'ast_partial:A':
        raise ValueError('AST-A required')
    meta = json.loads(bundle.components[COMPONENT].read_text())
    model = json.loads(bundle.components['model/ast_detector.json'].read_text())
    if meta['checkpoint_sha256'] != model['source_checkpoint_sha256'] or meta['rule'] != bundle.spec.decision_rule.to_dict():
        raise ValueError('Calibration/model identity or rule mismatch')
    if meta['score_type'] != ('calibrated_probability' if meta['accepted'] else 'model_score'):
        raise ValueError('Invalid score type')
    mapping(meta['method'], meta['parameters'], [.5])
    return bundle, meta


def package(root, run, destination):
    root, run, destination = Path(root), Path(run), Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    prospective = json.loads((run/'prospective.json').read_text())
    recipe = prospective['recipe']
    source = root/recipe['source_bundle']
    if digest(source/'bundle.json') != recipe['input_hashes'][recipe['source_bundle']+'/bundle.json']:
        raise ValueError('Source bundle identity changed')
    original = load_bundle(source)
    if original.spec.model_version != 'ast_partial:A':
        raise ValueError('AST-A required')
    result = json.loads((run/'operating.json').read_text())
    if digest(run/'fit.json') != result['fit_sha256']:
        raise ValueError('Fit changed')
    info = json.loads(original.components['model/ast_detector.json'].read_text())
    meta = {k:result[k] for k in ('method','parameters','accepted','score_type','rule')}
    meta.update(checkpoint_sha256=info['source_checkpoint_sha256'], recipe_sha256=prospective['recipe_sha256'],
                operating_sha256=digest(run/'operating.json'), reference_precision='cuda_fp16', status='candidate_pending_review_and_parity')
    spec = BundleSpecification(1, 'bundle:r4-n1-ast-A:'+run.name, 'ast_partial:A', original.spec.preprocessing_hash,
                               DecisionRule(**result['rule']), 'genuine', 'tts')
    write_bundle(destination, spec, {**original.components, COMPONENT:json.dumps(meta, allow_nan=False).encode()},
                 extra={'stage':'r4_n1_ast_candidate','locked':False,'promoted':False})
    inspect(destination)
    for name, path in original.components.items():
        if digest(path) != digest(destination/name):
            raise ValueError('Copied model bytes differ')
    return destination


def load_ast_a_bundle(path, *, device='cuda', precision='fp16', threshold=None):
    if threshold is not None or precision != 'fp16' or not str(device).startswith('cuda'):
        raise ValueError('Use the saved rule and reference CUDA FP16; other precision remains unverified')
    from echoproof.models.ast_adapter import ASTBundleAdapter
    from echoproof.inference import LoadedBundle
    bundle, meta = inspect(path)
    adapter = ASTBundleAdapter.load(path, device=device, precision=precision)
    return LoadedBundle(spec=bundle.spec, scorer=adapter.score_chunks,
                        calibrate=(lambda score:float(mapping(meta['method'],meta['parameters'],[score])[0])) if meta['accepted'] else None,
                        score_type=meta['score_type'])
