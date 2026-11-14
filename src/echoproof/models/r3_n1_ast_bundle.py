"""Explicit candidate loader; never silently opts the app into an unapproved AST."""
import json
from echoproof.bundle import load_bundle
from echoproof.evaluation.calibrate import apply_temperature


def load_ast_candidate(path, *, device='cuda', precision='fp16'):
    from echoproof.models.ast_adapter import ASTBundleAdapter
    from echoproof.inference import LoadedBundle
    bundle = load_bundle(path)
    meta = json.loads(bundle.components['calibration/ast.json'].read_text())
    if meta.get('schema') != 'r3_n1_ast_candidate_v1' or bundle.spec.model_version != 'ast_partial:B':
        raise ValueError('Incompatible AST candidate')
    if precision != 'fp16':
        raise ValueError('CPU/fp32 threshold parity has not been established for this candidate')
    use = meta['temperature_accepted_by_candidate_rule']
    if type(use) is not bool or meta['score_type'] != ('calibrated_probability' if use else 'model_score'):
        raise ValueError('Invalid calibration semantics')
    apply_temperature([0.5], meta['temperature'], epsilon=meta['epsilon'])
    adapter = ASTBundleAdapter.load(path, device=device, precision=precision)
    calibrate = (lambda x: apply_temperature([x], meta['temperature'], epsilon=meta['epsilon'])[0]) if use else None
    return LoadedBundle(spec=bundle.spec, scorer=adapter.score_chunks, calibrate=calibrate, score_type=meta['score_type'])
