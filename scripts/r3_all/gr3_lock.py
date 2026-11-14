"""Build the GR3 (R3-ALL) Protocol B lock record for the four candidates. Default is a dry run; --write creates the file and refuses to overwrite.
Does not touch locks/B_lock.json (the provisional GR2 LR lock).
Usage: python scripts/r3_all/gr3_lock.py [--write --out locks/B_lock_gr3.json]"""
import argparse, hashlib, json, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, 'src')
from echoproof.bundle import load_bundle

CANDIDATES = {
    'LR': 'runs/r3/r3_n1_lr/selected_lr_v1/handoff/bundle',
    'BiGRU': 'runs/r3/r3_n1_bigru/selected_bigru_v1/handoff/bundle',
    'WavLM': 'bundles/r3_n1_wavlm_head_B_calibrated',
    'AST': 'runs/r3/r3_n1_ast_platt_candidates/20261114T150254Z-046f980d/bundle',
}
LAT = {'LR': 'reports/r3/r3_n2_local_lr_latency_20261114T135646Z.json', 'BiGRU': 'reports/r3/r3_n2_local_bigru_latency_20261114T140417Z.json',
       'WavLM': 'reports/r3/r3_n2_local_wavlm_latency_20261114T133632Z.json'}
MEM = {m: f'reports/r3/gr3/memory_{m.lower()}_20261114.json' for m in ('LR', 'BiGRU', 'WavLM')}
SELECTION = 'reports/r3/gr3/selection_20261114.json'
COMPARE = 'reports/r3/model_comparison/compare.json'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def git(*a):
    return subprocess.run(['git', *a], capture_output=True, text=True, check=True).stdout.strip()


def candidate(name, path):
    b = load_bundle(path)  # verifies component hashes and metadata
    extra = b.raw_metadata.get('extra', {})
    rec = {'bundle_path': path, 'bundle_id': b.spec.bundle_id, 'model_version': b.spec.model_version, 'bundle_json_sha256': sha(Path(path) / 'bundle.json'),
           'preprocessing_hash': b.spec.preprocessing_hash, 'decision_rule': b.spec.decision_rule.to_dict(),
           'components': {k: {'sha256': h, 'bytes': Path(b.components[k]).stat().st_size} for k, h in sorted(b.component_hashes.items())},
           'bundle_extra': {k: extra[k] for k in ('stage', 'review_status', 'deployment_ready', 'officially_promoted', 'promoted', 'locked', 'score_type') if k in extra}}
    if name in LAT:
        d = json.load(open(LAT[name])); g = d['groups']['10']; t = g['summary_ns']['total']; f = g['fresh_process_summary_ns']
        assert d['bundle']['bundle_json_sha256'] == rec['bundle_json_sha256'], f'{name}: latency report is for a different bundle'
        rec['latency_local_10s'] = {'report': LAT[name], 'report_sha256': sha(LAT[name]), 'warm_p50_ms': round(t['p50'] / 1e6, 1), 'warm_p95_ms': round(t['p95'] / 1e6, 1),
                                    'timed_calls': t['count'], 'parity_exact': bool(g['parity']['reference_match'] and not g['parity']['decision_flips']),
                                    'cold_process_start_to_first_result_p50_s': round(f['process_to_first_prediction_ns']['p50'] / 1e9, 1)}
        m = json.load(open(MEM[name]))
        rec['memory_local'] = {'report': MEM[name], 'report_sha256': sha(MEM[name]), 'peak_working_set_mb_after_60s_clip': m['after_60s_clip']['peak_working_set_mb']}
    else:
        rec['latency_local_10s'] = None
        rec['memory_local'] = None
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--write', action='store_true')
    ap.add_argument('--out', default='locks/B_lock_gr3.json')
    a = ap.parse_args()
    sel = json.load(open(SELECTION))
    gr2 = json.load(open('locks/B_lock.json'))
    cands = {n: candidate(n, p) for n, p in CANDIDATES.items()}
    for n in cands:
        cands[n]['selection_panel'] = {'unseen_xtts': sel['candidates'][n]['unseen_xtts'], 'known_melo_vits': sel['candidates'][n]['known_melo_vits']}
        cands[n]['rank'] = sel['ranking'].index(n) + 1
    status = git('status', '--porcelain')
    src = {str(p).replace('\\', '/'): sha(p) for p in sorted(Path('src/echoproof').rglob('*.py'))}
    top = sel['ranking'][0]
    rec = {
        'schema_version': 1, 'lock_id': 'lock:protocol_b_gr3_v1', 'protocol': 'B', 'created_utc': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        'supersedes_note': 'locks/B_lock.json (provisional GR2 LR lock) is kept unchanged as history.',
        'split': {'split_hash': gr2['split']['split_hash'], 'split_version': gr2['split']['split_version'], 'source': 'locks/B_lock.json'},
        'selection_rule': sel['rule'], 'selection_receipt': {'path': SELECTION, 'sha256': sha(SELECTION), 'ranking': sel['ranking'], 'tie_at_top': sel['tie_at_top']},
        'candidates': cands,
        'production_selection': {'bundle_id': cands[top]['bundle_id'], 'model': top,
                                 'basis': 'highest unseen-XTTS selection macro F1 at raw 0.5 among the four candidates; no tie'},
        'policy_decisions_user_reported': {
            'date': '2026-11-14', 'by': 'project owner (user-reported; no team or individual approval record in the repository)',
            'thresholds_and_policy': 'every bundle locked as finalised: macro-F1 threshold chosen on the operating role, no FPR cap, no prevalence correction, no abstention, no amendment',
            'demo_device': "Subhan's HP (HP EliteBook 840 G7 Notebook PC, Intel Core i5-10310U, 15.8 GB, Windows 11), CPU only, fp32, torch 4 threads. Named by the project owner on 2026-11-14 AFTER the timing runs; the contract asks to name it before timing. OPEN-004 is closed by this naming, not by a pre-timing declaration. The project owner reports that the team reviewed and accepted the name (user-reported; no team record in the repository).",
            'five_second_target_reading': 'warm p95 on the 10 s clip; cold start not included (all measured models exceed 5 s if it is)'},
        'deployment_qualification': {
            'device': 'Intel Core i5-10310U, 4 cores/8 logical, 15.8 GB, Windows 11, CPU, fp32, 4 torch threads; 5 warm-up calls (3 for WavLM), 100 timed calls (30 for WavLM), batch 1',
            'checks': {
                'bundle_loads_and_hashes_verified': {n: True for n in cands},
                'finite_complete_prediction_and_parity': {'LR': True, 'BiGRU': True, 'WavLM': True,
                                                          'AST': 'not run locally (CPU fp32 not verified; recipient T4/fp16 sample passed, review pending)'},
                'input_limits': 'enforced by the shared analyze() before any model runs (size and duration limits); model-independent; covered by tests/test_r1_n2_inference.py; not re-run for this lock',
                'memory_feasibility': {n: (cands[n]['memory_local'] or 'not measured') for n in cands},
                'five_second_target_warm_p95_10s': {n: (cands[n]['latency_local_10s']['warm_p95_ms'] if cands[n]['latency_local_10s'] else 'not measured') for n in cands}},
            'result_for_production_candidate': 'WavLM: warm p95 within 5 s on the named laptop; qualification evidence is local and development-grade (see limitations)'},
        'code': {'git_head': git('rev-parse', 'HEAD'), 'working_tree_dirty': bool(status), 'uncommitted_entries_sha256': hashlib.sha256(status.encode()).hexdigest(),
                 'src_echoproof_sha256': src, 'scripts_sha256': {str(p).replace('\\', '/'): sha(p) for p in sorted(Path('scripts/r3_all').glob('*.py'))},
                 'python': sys.version.split()[0]},
        'comparison_report': {'path': COMPARE, 'sha256': sha(COMPARE), 'human_readable': 'REPORT.md'},
        'test_access_audit': {'audit_status': 'partially_verified', 'test_partition_accessed': False,
                              'note': 'The comparison, calibration review and latency work used only the selection, calibration and operating roles and fixtures. Earlier work by other owners was not re-audited; filesystem access was not monitored (as in the GR2 lock).'},
        'limitations': [
            'All four bundles are development bundles: none is independently reviewed; BiGRU review status is pending; AST calibration deviates from the contract method on team authority reported by its owner; WavLM calibration finalised by the project owner without a team record.',
            'Thresholds were chosen on the operating role, so operating metrics are development results, not independent test performance.',
            'WavLM at its locked threshold flags 49 of 260 genuine operating recordings (18.8%); the policy assumes 75% TTS prevalence.',
            'The demo device was named after timing. Latency is warm, CPU, one laptop; AST latency, memory and CPU fp32 decision parity were not measured.',
            'AST is verified on CUDA/fp16 only, on a 12-recording sample; its wrapper refuses CPU.',
            'The working tree was not clean when the record was built; uncommitted entries are hashed above.']}
    text = json.dumps(rec, indent=2, sort_keys=True)
    print(f"lock {rec['lock_id']}: production {top} ({rec['production_selection']['bundle_id']}); ranking {sel['ranking']}; dirty tree {rec['code']['working_tree_dirty']}; {len(text)} bytes")
    if a.write:
        out = Path(a.out)
        if out.exists():
            raise SystemExit(f'refusing to overwrite {out}')
        out.write_text(text + '\n', encoding='utf-8', newline='\n')
        print('written', out, sha(out))


main()
