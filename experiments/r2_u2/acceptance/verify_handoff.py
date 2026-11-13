"""Read-only recipient verification. No downloads, inference, training or publication.

Lives outside the experiment's source-discovery directories deliberately.
The optional CPU check loads only model weights into a fresh AST wrapper and
creates an empty optimizer; it never restores the warm-up optimizer or scaler.
"""
import argparse
import contextlib
import csv
import hashlib
import json
import os
from pathlib import Path
import sys


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checked_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    require(not Path(relative).is_absolute() and path.is_relative_to(root) and path != root,
            'Artifact path escapes project root: ' + relative)
    return path


def verify_files(root, files):
    for name, identity in files.items():
        path = checked_path(root, name)
        require(path.is_file(), 'Missing required artifact: ' + str(path))
        require(path.stat().st_size == identity['bytes'] and sha(path) == identity['sha256'],
                'Artifact identity mismatch: ' + name)


def verify_ledger(root, contract):
    # Bind the row, not the mutable team ledger file.
    for relative in contract['locations']:
        with checked_path(root, relative).open(newline='', encoding='utf-8-sig') as stream:
            rows = [r for r in csv.DictReader(stream) if r['run_id'] == contract['row']['run_id']]
        require(rows == [contract['row']], 'Missing, duplicate or changed ledger row: ' + relative)


# Manifest entries deliberately NOT identity-checked by this verifier. Each is named here,
# reported in the output as waived, and the manifest itself is not changed.
# docs/r2_u2_acceptance.md: the manifest binds an earlier version of the guide that no
# committed or archived copy reproduces (checked across all 27 commits and all git
# objects). The waiver is by the project owner's instruction. It does not validate that file.
WAIVED_FILES = ('docs/r2_u2_acceptance.md',)


def verify(root, manifest):
    waived = [name for name in WAIVED_FILES if name in manifest['files']]
    verify_files(root, {name: identity for name, identity in manifest['files'].items() if name not in WAIVED_FILES})
    verify_ledger(root, manifest['ledger'])
    result = json.loads(checked_path(root, manifest['result']).read_text())
    require(result['experimental_checks_passed'] is True and result['research_accepted'] is False,
            'Historical experimental disposition changed')
    require(result['summary']['best_epoch'] == 2, 'Unexpected selected epoch')
    return {'status': 'artifact_identity_verified_recipient_load_pending',
            'manifest_sha256': sha(Path(__file__).with_name('manifest.json')),
            'verified_files': len(manifest['files']) - len(waived), 'waived_files': waived,
            'run_id': manifest['run_id'],
            'checkpoint_sha256': manifest['files'][manifest['initializer']]['sha256'],
            'recipient_model_load_verified': False, 'recipient_access_verified': False,
            'note': 'Local readability does not establish another teammate\'s Drive permission.'}


def load_check(root, manifest, snapshot):
    # Intentionally no automatic package installation or Hub retrieval.
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import importlib.metadata
    import torch
    require(importlib.metadata.version('transformers') == '5.22.0', 'Use Transformers 5.22.0')
    require(importlib.metadata.version('torch') == '2.14.1+cu130', 'Use the recorded Account B torch stack')
    sys.path.insert(0, str(Path(root).resolve() / 'src'))
    from echoproof.models.ast_detector import ASTDetector
    reference = json.loads(checked_path(root, 'experiments/r2_u2/configs/model_reference.json').read_text())
    snapshot = Path(snapshot).resolve()
    require(snapshot.is_dir() and snapshot.name == reference['model']['revision'], 'Wrong/missing local AST snapshot')
    require(sha(snapshot / 'model.safetensors') == reference['model']['weights_sha256'], 'Pretrained weights differ')
    require(sha(snapshot / 'preprocessor_config.json') == reference['model']['processor_file_sha256'], 'Processor differs')
    checkpoint_path = checked_path(root, manifest['initializer'])
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    require(checkpoint['epoch'] == 2 and checkpoint['config_hash'] == manifest['config_hash'], 'Wrong checkpoint metadata')
    require(all(torch.isfinite(t).all().item() for t in checkpoint['model'].values()), 'Non-finite model tensor')
    model = ASTDetector.from_dir(snapshot, device='cpu', seed=42)
    require(model.digests() == reference['initial_group_sha256'], 'Fresh model identity differs')
    model.load_state_dict(checkpoint['model'], strict=True)
    expected = json.loads(checked_path(root, manifest['result']).read_text())['freeze_proofs']['best']['group_sha256']
    require(model.digests() == expected, 'Loaded model fingerprints differ')
    model.model.config.id2label = {0: 'genuine', 1: 'tts'}
    model.model.config.label2id = {'genuine': 0, 'tts': 1}
    model.set_stage('partial', n_blocks=2)
    require(model.n_blocks == 12, 'Unexpected AST block layout')
    expected_groups = {'block.10', 'block.11', 'final_norm', 'classifier'}
    for name, param in model.model.named_parameters():
        require(param.requires_grad == (model._group(name) in expected_groups), 'Wrong trainable flag: ' + name)
    groups = model.param_groups(head_lr=1e-4, backbone_lr=1e-5, weight_decay=.01)
    optimizer = torch.optim.AdamW(groups, lr=1e-4, weight_decay=.01)
    optimized = [id(p) for group in optimizer.param_groups for p in group['params']]
    trainable = [id(p) for p in model.parameters() if p.requires_grad]
    require(len(optimized) == len(set(optimized)) and set(optimized) == set(trainable), 'Optimizer coverage differs')
    require({g['name']: g['lr'] for g in optimizer.param_groups} == {'head': 1e-4, 'backbone': 1e-5}, 'Wrong group learning rates')
    require(all(g['weight_decay'] == .01 for g in optimizer.param_groups) and not optimizer.state, 'Optimizer is not fresh')
    require(model.digests() == expected and sha(checkpoint_path) == manifest['files'][manifest['initializer']]['sha256'],
            'Weights changed during load check')
    return {'recipient_model_load_verified': True, 'status': 'artifact_identity_and_cpu_load_verified',
            'device': 'cpu', 'torch': torch.__version__, 'transformers': importlib.metadata.version('transformers'),
            'trainable_groups': sorted(expected_groups), 'trainable_tensors': len(trainable),
            'optimizer_state_entries': len(optimizer.state), 'optimizer_updates': 0,
            'resume_used': False, 'inference_run': False, 'r3_training_authorized_by_check': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', required=True)
    parser.add_argument('--model-snapshot', help='Optional existing pinned AST snapshot; enables CPU-only load check')
    args = parser.parse_args()
    try:
        manifest = json.loads(Path(__file__).with_name('manifest.json').read_text())
        report = verify(args.project_root, manifest)
        if args.model_snapshot:
            with contextlib.redirect_stdout(sys.stderr):
                report.update(load_check(args.project_root, manifest, args.model_snapshot))
    except Exception as exc:
        print(json.dumps({'status': 'verification_failed', 'error': str(exc), 'error_type': type(exc).__name__}, indent=2))
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
