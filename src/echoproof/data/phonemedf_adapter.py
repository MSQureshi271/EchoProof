"""Convert a verified inventory into a provisional PhonemeDF metadata snapshot."""
from __future__ import annotations
import hashlib
import json
import os
import platform
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from ..contracts import ContractError, Recording
from .inventory import _digest, _json

COMPONENTS = {
    'genuine': ('genuine', 0, 'genuine', None),
    'xtts': ('XTTSv2', 1, 'tts', 'XTTSv2'),
    'chatterbox': ('ChatterboxTTS', 1, 'tts', 'ChatterboxTTS'),
    'melo': ('MeloTTS', 1, 'tts', 'MeloTTS'),
    'vits': ('VITS', 1, 'tts', 'VITS'),
}


def adapt_row(row, config, snapshot_revision):
    """Return a provisional Recording or an explicit disposition for one inventory row."""
    component = row.get('component')
    file_id = row.get('file_id')
    if component == 'transcripts':
        return None, {'file_id':file_id, 'component':component, 'state':'pending', 'reason':'transcript_for_pairing'}
    if component not in COMPONENTS:
        return None, {'file_id':file_id, 'component':component, 'state':'pending', 'reason':'unsupported_component'}
    if row.get('state') != 'completed' or row.get('file_type') != '.wav':
        return None, {'file_id':file_id, 'component':component, 'state':'pending', 'reason':'inventory_incomplete_or_non_wav'}
    try:
        p = PurePosixPath(row['relative_path'])
        if p.is_absolute() or any(part in ('', '.', '..') for part in row['relative_path'].split('/')) or '\\' in row['relative_path']:
            raise ValueError('unsafe relative path')
        _, label, method, family = COMPONENTS[component]
        record = Recording(schema_version=1, recording_id=file_id,
            original_path=(PurePosixPath(config['sources'][component]) / p).as_posix(),
            dataset='phonemedf', dataset_revision=snapshot_revision,
            label=label, label_verified=False, method=method, method_verified=False,
            generator_family=family, generator_version=None, language=None,
            source_utterance_id=None, source_speaker_id=None, target_voice_id=None,
            reference_id=None, transcript_hash=None, duration_seconds=None,
            sample_rate=None, channels=None, codec=None, sha256=row['sha256'],
            quality_flags=[], retained=False, exclusion_reason='pending_source_admission', license_ref=None)
    except (KeyError, TypeError, ValueError, ContractError) as exc:
        return None, {'file_id':file_id, 'component':component, 'state':'pending', 'reason':'invalid_record', 'detail':str(exc)}
    return record, {'file_id':file_id, 'component':component, 'state':'pending', 'reason':'pending_source_admission'}


def create_snapshot(root, config, config_path, inventory_dir):
    import pyarrow as pa
    import pyarrow.parquet as pq
    root = Path(root).resolve(); inventory_dir = Path(inventory_dir).resolve()
    allowed = (root / config['outputs']['inventory']).resolve()
    if not inventory_dir.is_relative_to(allowed) or inventory_dir == allowed:
        raise ValueError('inventory must be a run within the configured inventory root')
    summary_path = inventory_dir / 'scan_summary.json'
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    if summary.get('status') != 'complete' or any(summary['counts'].get(k) for k in ('error','pending','unstable')):
        raise ValueError('inventory has unresolved errors')
    for name, expected in summary['artifacts_sha256'].items():
        if Path(name).name != name or _digest(inventory_dir / name) != expected:
            raise ValueError('inventory artifact hash mismatch: ' + name)
    registry_path = root / config['source_registry']
    if summary.get('fingerprint', {}).get('config_sha256') != _digest(config_path):
        raise ValueError('inventory configuration differs from current configuration')
    if summary.get('fingerprint', {}).get('registry_sha256') != _digest(registry_path):
        raise ValueError('inventory source registry differs from current registry')
    registry = json.loads(registry_path.read_text(encoding='utf-8'))
    source = next(x for x in registry['sources'] if x['source_id'] == 'phonemedf')
    registered = {entry['component']:entry for entry in source['components']}
    for component, (name, _, method, _) in COMPONENTS.items():
        entry = registered.get(name)
        if entry is None or entry.get('path') != config['sources'][component] or entry.get('publisher_method') != method:
            raise ValueError('source registry conflicts with component mapping: ' + component)
    revision = source['local_snapshot_id']
    outbase = (root / 'data/manifests/phase2_raw').resolve()
    if not outbase.is_relative_to(root): raise ValueError('snapshot output escapes project root')
    outbase.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:12]
    out = outbase / run_id; out.mkdir(exist_ok=False)
    counts = {'rows_seen':0,'recordings':0,'dispositions':0,'invalid':0}
    seen = set(); batch = []
    schema = pa.schema([('schema_name',pa.string()),('schema_version',pa.int64()),('payload_json',pa.string())])
    writer = pq.ParquetWriter(out/'raw_recordings.parquet', schema)
    try:
        with (inventory_dir/'file_inventory.jsonl').open(encoding='utf-8') as inp, (out/'dispositions.jsonl').open('x',encoding='utf-8',newline='\n') as dispositions, (out/'mapping_evidence.jsonl').open('x',encoding='utf-8',newline='\n') as mappings:
            for line in inp:
                row = json.loads(line); counts['rows_seen'] += 1
                file_id = row.get('file_id')
                if not isinstance(file_id, str) or file_id in seen:
                    raise ValueError('missing or duplicate inventory file ID')
                seen.add(file_id)
                record, disposition = adapt_row(row, config, revision)
                dispositions.write(_json(disposition)+'\n'); counts['dispositions'] += 1
                if record is None:
                    if disposition['reason'] == 'invalid_record': counts['invalid'] += 1
                    continue
                mappings.write(_json({'file_id':file_id,'component':row['component'],'candidate_label':record.label,
                    'candidate_method':record.method,'candidate_family':record.generator_family,
                    'label_verified':False,'method_verified':False,'evidence':'configs/data_sources.json',
                    'source_path':row['relative_path']})+'\n')
                batch.append({'schema_name':'recording','schema_version':1,'payload_json':record.to_json()})
                counts['recordings'] += 1
                if len(batch) >= 2048:
                    writer.write_table(pa.Table.from_pylist(batch,schema=schema)); batch.clear()
            if batch: writer.write_table(pa.Table.from_pylist(batch,schema=schema))
            for stream in (dispositions,mappings): stream.flush(); os.fsync(stream.fileno())
    finally:
        writer.close()
    if counts['rows_seen'] != sum(summary['counts'].values()): raise ValueError('inventory row count differs from summary')
    if _digest(inventory_dir/'file_inventory.jsonl') != summary['artifacts_sha256']['file_inventory.jsonl']:
        raise ValueError('inventory changed during snapshot creation')
    provenance = {'source':'phonemedf','local_snapshot_id':revision,'publisher_release':source.get('publisher_candidate'),
        'publisher_revision_basis':'user_attestation','permissions':'user_attested_permitted',
        'input_inventory':str(inventory_dir.relative_to(root)), 'inventory_summary_sha256':_digest(summary_path),
        'inventory_content_digest':summary['content_digest_sha256'], 'registry_sha256':_digest(registry_path),
        'config_sha256':_digest(config_path), 'adapter_sha256':_digest(Path(__file__)),
        'contracts_sha256':_digest(Path(__file__).parents[1]/'contracts.py'),
        'python':sys.version,'platform':platform.platform(),'pyarrow':pa.__version__,
        'input_limitation':'Retained inventory hashes were obtained before the user-reported duplicate cleanup; current audio bytes have not been rehashed.',
        'admission':'pending','partitions_created':False}
    (out/'provenance.json').write_text(_json(provenance)+'\n',encoding='utf-8')
    artifact_names = ['raw_recordings.parquet','dispositions.jsonl','mapping_evidence.jsonl','provenance.json']
    manifest = {'schema_version':1,'task':'P2-05','run_id':run_id,'status':'ready_for_review' if not counts['invalid'] else 'needs_attention',
        'counts':counts,'artifacts_sha256':{name:_digest(out/name) for name in artifact_names}}
    (out/'snapshot_manifest.json').write_text(_json(manifest)+'\n',encoding='utf-8')
    return {'snapshot':str(out),**manifest}
