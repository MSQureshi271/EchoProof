"""P2-07--09 header inspection, reconciliation and capacity planning."""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import sqlite3
import struct
import time
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from .inventory import _digest, _json, _write_jsonl

COMPONENTS = ('genuine', 'xtts', 'chatterbox', 'melo', 'vits')


def _now():
    return datetime.now(timezone.utc).isoformat()


def _write(path, obj):
    temp = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temp.open('x', encoding='utf8', newline='\n') as f:
        f.write(_json(obj) + '\n'); f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)


def _source(root, relative):
    value = PurePosixPath(relative)
    if value.is_absolute() or '\\' in relative or any(p in ('', '.', '..') for p in relative.split('/')):
        raise ValueError('unsafe source path')
    path = (root / Path(*value.parts)).resolve()
    if not path.is_relative_to((root / 'dataset').resolve()): raise ValueError('source outside dataset')
    return path


def _component(path, config):
    path = PurePosixPath(path)
    matches = [c for c in COMPONENTS if path.is_relative_to(PurePosixPath(config['sources'][c]))]
    if len(matches) != 1: raise ValueError('recording path does not identify one audio component')
    return matches[0]


def _riff_check(path):
    """Reject a WAV whose RIFF container declares bytes past the actual file."""
    size = path.stat().st_size
    with path.open('rb') as stream: head = stream.read(12)
    if len(head) < 12: raise ValueError('truncated container header')
    if head[:4] == b'RIFF' and head[8:12] == b'WAVE':
        if struct.unpack('<I', head[4:8])[0] + 8 > size: raise ValueError('truncated RIFF container')


def inspect_header(path, file_id, component, relative):
    import soundfile as sf
    row = {'file_id':file_id,'component':component,'path':relative,'state':'header_error'}
    try:
        _riff_check(path)
        info = sf.info(str(path))
        if info.samplerate <= 0 or info.channels <= 0 or info.frames <= 0: raise ValueError('invalid or empty audio header')
        row.update(state='ok', sample_rate=info.samplerate, channels=info.channels,
                   frames=info.frames, duration_seconds=info.frames/info.samplerate,
                   format=info.format, subtype=info.subtype, failure=None)
    except (OSError, RuntimeError, ValueError) as exc:
        row['failure'] = f'{type(exc).__name__}: {exc}'
    return row


def decode_sample(path, header, max_seconds, block_frames=65536, max_channels=32, max_frames=20000000):
    import numpy as np
    import soundfile as sf
    result = {'file_id':header['file_id'],'component':header['component'],'path':header['path'],'state':'decode_error'}
    if header['state'] != 'ok':
        result['failure'] = 'header_error'; return result
    if header['duration_seconds'] > max_seconds or header['channels'] > max_channels or header['frames'] > max_frames:
        result.update(state='resource_limit', failure='decode duration, frame or channel budget exceeded'); return result
    try:
        _riff_check(path)
        total=0; peak=0.; sumsq=0.; samples=0
        with sf.SoundFile(str(path)) as audio:
            if (audio.samplerate,audio.channels,len(audio)) != (header['sample_rate'],header['channels'],header['frames']):
                raise ValueError('header changed since inspection')
            while True:
                block=audio.read(block_frames,dtype='float32',always_2d=True)
                if not len(block): break
                if not np.isfinite(block).all(): raise ValueError('non-finite samples')
                total+=len(block); samples+=block.size
                peak=max(peak,float(np.max(np.abs(block))))
                sumsq+=float(np.sum(block.astype('float64')**2))
        if total != header['frames']: raise ValueError('decoded frame count differs from header')
        result.update(state='ok',decoded_frames=total,peak=peak,rms=math.sqrt(sumsq/samples),failure=None)
    except (OSError,RuntimeError,ValueError) as exc:
        result['failure']=f'{type(exc).__name__}: {exc}'
    return result


def _check_inputs(root, snapshot, pairs):
    snapshot=Path(snapshot).resolve();pairs=Path(pairs).resolve()
    for p,b in ((snapshot,'phase2_raw'),(pairs,'phase2_pairs')):
        boundary=(root/'data/manifests'/b).resolve()
        if not p.is_relative_to(boundary) or p==boundary: raise ValueError('input outside expected snapshot root')
    m=json.loads((snapshot/'snapshot_manifest.json').read_text(encoding='utf8'))
    q=json.loads((pairs/'pairing_summary.json').read_text(encoding='utf8'))
    if m['status']!='ready_for_review' or q['status']!='ready_for_review': raise ValueError('input not ready for review')
    if q['snapshot_manifest_sha256']!=_digest(snapshot/'snapshot_manifest.json'): raise ValueError('pairing references different snapshot')
    for folder,manifest in ((snapshot,m),(pairs,q)):
        for name,digest in manifest['artifacts_sha256'].items():
            if Path(name).name!=name or _digest(folder/name)!=digest: raise ValueError('input hash mismatch: '+name)
    return m,q


def inspect(root, config, snapshot, pairs, resume=None):
    import pyarrow.parquet as pq
    import pyarrow
    import soundfile
    import numpy
    import sys
    from ..contracts import Recording
    root=Path(root).resolve();snapshot=Path(snapshot).resolve();pairs=Path(pairs).resolve()
    manifest,pair_summary=_check_inputs(root,snapshot,pairs)
    output=(root/'data/manifests/phase2_inspection').resolve()
    if not output.is_relative_to(root): raise ValueError('unsafe output root')
    output.mkdir(parents=True,exist_ok=True)
    run_id=resume or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:12]
    if not run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-' for c in run_id): raise ValueError('invalid run ID')
    run=output/run_id
    if resume:
        if not (run/'checkpoint.sqlite').is_file(): raise ValueError('checkpoint missing')
        if (run/'inspection_summary.json').exists(): raise ValueError('inspection already finished; start a new run')
    else: run.mkdir(exist_ok=False)
    print('P2-07 run ID:',run_id,flush=True)
    identity={'snapshot_manifest_sha256':_digest(snapshot/'snapshot_manifest.json'),
              'pairing_summary_sha256':_digest(pairs/'pairing_summary.json'),
              'config_sha256':hashlib.sha256(_json(config).encode()).hexdigest(),
              'registry_sha256':_digest(root/config['source_registry']),
              'contracts_sha256':_digest(Path(__file__).parents[1]/'contracts.py'),
              'code_sha256':_digest(Path(__file__)),'root':str(root),'python':sys.version,
              'pyarrow':pyarrow.__version__,'soundfile':soundfile.__version__,'numpy':numpy.__version__}
    lock=run/'running.lock'
    try: fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    except FileExistsError as exc: raise RuntimeError('inspection locked; ensure previous runtime is stopped') from exc
    os.close(fd);db=None
    try:
        db=sqlite3.connect(run/'checkpoint.sqlite'); db.execute('PRAGMA synchronous=FULL')
        db.execute('CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY,value TEXT NOT NULL)')
        db.execute('CREATE TABLE IF NOT EXISTS headers (id TEXT PRIMARY KEY, component TEXT NOT NULL, row_json TEXT NOT NULL, sha256 TEXT NOT NULL)')
        if db.execute('PRAGMA integrity_check').fetchone()[0]!='ok': raise ValueError('checkpoint corruption')
        old=db.execute("SELECT value FROM meta WHERE key='identity'").fetchone()
        if old and json.loads(old[0])!=identity: raise ValueError('resume inputs or code changed')
        if resume and not old: raise ValueError('missing checkpoint identity')
        if not old: db.execute("INSERT INTO meta VALUES ('identity',?)",(_json(identity),));db.commit()
        for raw,sha in db.execute('SELECT row_json,sha256 FROM headers'):
            if hashlib.sha256(raw.encode()).hexdigest()!=sha: raise ValueError('checkpoint row hash mismatch')
        _write(run/'run_state.json',{'status':'running','run_id':run_id,'updated_utc':_now()})
        start=time.monotonic();processed=0;new=0
        pf=pq.ParquetFile(snapshot/'raw_recordings.parquet')
        if pf.schema_arrow.names != ['schema_name','schema_version','payload_json']: raise ValueError('unexpected parquet envelope')
        for batch in pf.iter_batches(batch_size=2048):
            for entry in batch.to_pylist():
                if entry['schema_name']!='recording' or entry['schema_version']!=1: raise ValueError('invalid parquet envelope')
                rec=Recording.from_json(entry['payload_json']);processed+=1
                if db.execute('SELECT 1 FROM headers WHERE id=?',(rec.recording_id,)).fetchone(): continue
                component=_component(rec.original_path,config)
                try:
                    path=_source(root,rec.original_path)
                    row=inspect_header(path,rec.recording_id,component,rec.original_path)
                except (ValueError,OSError) as exc:
                    row={'file_id':rec.recording_id,'component':component,'path':rec.original_path,
                         'state':'header_error','failure':f'{type(exc).__name__}: {exc}'}
                raw=_json(row);db.execute('INSERT INTO headers VALUES (?,?,?,?)',(rec.recording_id,component,raw,hashlib.sha256(raw.encode()).hexdigest()))
                new+=1
                if new%config['checkpoint_interval']==0:
                    db.commit();print(f'P2-07 headers processed: {processed}/{manifest["counts"]["recordings"]}',flush=True)
        db.commit()
        if processed!=manifest['counts']['recordings'] or db.execute('SELECT COUNT(*) FROM headers').fetchone()[0]!=processed: raise ValueError('header count mismatch')
        unusual_stems=set()
        with (pairs/'coverage.jsonl').open(encoding='utf8') as inp:
            for line in inp:
                coverage=json.loads(line)
                if any(n!=1 for n in coverage['counts'].values()): unusual_stems.add(coverage['candidate_stem'])
        unmatched=set()
        with (pairs/'pair_dispositions.jsonl').open(encoding='utf8') as inp:
            for line in inp:
                relation=json.loads(line)
                if not relation['has_genuine'] or not relation['has_transcript'] or relation['candidate_stem'] in unusual_stems:
                    unmatched.add(relation['recording_id'])
        # Allocate a deterministic balanced sample across component and duration/header strata.
        budget=config['sample_count'];seed=config['sample_seed']
        per_component={c:budget//len(COMPONENTS)+(i<budget%len(COMPONENTS)) for i,c in enumerate(COMPONENTS)}
        selected=[]; header_counts=Counter();component_counts=defaultdict(Counter)
        for comp in COMPONENTS:
            rows=[json.loads(r[0]) for r in db.execute('SELECT row_json FROM headers WHERE component=? ORDER BY id',(comp,))]
            buckets=defaultdict(list)
            for row in rows:
                header_counts[row['state']]+=1;component_counts[comp][row['state']]+=1
                stratum='header_error' if row['state']!='ok' else ('unmatched' if row['file_id'] in unmatched else ('short_under_4s' if row['duration_seconds']<4 else 'long_4s_or_more'))
                row['stratum']=stratum
                rank=hashlib.sha256(f'{seed}:{row["file_id"]}'.encode()).hexdigest()
                buckets[stratum].append((rank,row))
            for values in buckets.values(): values.sort(key=lambda x:x[0])
            # Round-robin prevents the largest stratum from masking rare failures.
            while len([r for r in selected if r['component']==comp]) < per_component[comp]:
                advanced=False
                for key in ('header_error','unmatched','short_under_4s','long_4s_or_more'):
                    if len([r for r in selected if r['component']==comp]) >= per_component[comp]: break
                    if buckets[key]: selected.append(buckets[key].pop(0)[1]);advanced=True
                if not advanced: break
        hashes={}
        hashes['headers.jsonl']=_write_jsonl(run/'headers.jsonl',(json.loads(r[0]) for r in db.execute('SELECT row_json FROM headers ORDER BY component,id')))
        hashes['sample_selection.jsonl']=_write_jsonl(run/'sample_selection.jsonl',selected)
        decoded=[]
        for row in selected:
            try:
                path=_source(root,row['path']) if row['state']=='ok' else None
                decoded.append(decode_sample(path,row,config['decode_limit_seconds']))
            except (ValueError,OSError) as exc:
                decoded.append({'file_id':row['file_id'],'component':row['component'],'path':row['path'],
                                'state':'decode_error','failure':f'{type(exc).__name__}: {exc}'})
        hashes['sample_decodes.jsonl']=_write_jsonl(run/'sample_decodes.jsonl',decoded)
        outcomes=Counter(r['state'] for r in decoded)
        source=json.loads((root/config['source_registry']).read_text(encoding='utf8'))
        phonemedf=next(s for s in source['sources'] if s['source_id']=='phonemedf')
        admission={'status':'pending','basis':'administrative inspection only','user_reports_identity_review':True,
                   'identity_review_note':'User reports relationships and identities reviewed; P2-06 artifact still records candidate edges and unknown identities.',
                   'components':{c:{'status':'pending','header_counts':dict(component_counts[c]),
                     'reason':'source/method/label admission and review evidence required'} for c in COMPONENTS},
                   'publisher_revision_attested':phonemedf.get('publisher_revision_verified',False),
                   'audio_content_rehashed_after_cleanup':False}
        _write(run/'admission.json',admission);hashes['admission.json']=_digest(run/'admission.json')
        missing_components=[c for c in COMPONENTS if not sum(component_counts[c].values())]
        sample_by_component=dict(Counter(r['component'] for r in selected))
        sample_by_stratum=dict(Counter(r['stratum'] for r in selected))
        result={'task':'P2-07','run_id':run_id,'status':'ready_for_review' if not missing_components and len(selected)==min(budget,processed) and header_counts['header_error']==0 and not outcomes['decode_error'] and not outcomes['resource_limit'] else 'needs_attention',
                'header_counts':dict(header_counts),'component_header_counts':{c:dict(component_counts[c]) for c in COMPONENTS},
                'sample_count':len(selected),'sample_budget':budget,'sample_seed':seed,'sample_outcomes':dict(outcomes),
                'sample_by_component':sample_by_component,'sample_by_stratum':sample_by_stratum,'missing_components':missing_components,
                'sampling':'balanced component and header/duration/unmatched strata; sha256(seed:file_id) rank; administrative use only',
                'sample_limit_seconds':config['decode_limit_seconds'],'pairing_summary_sha256':identity['pairing_summary_sha256'],
                'provenance':identity,
                'elapsed_seconds_current_session':round(time.monotonic()-start,2),'artifacts_sha256':hashes,
                'unmatched_recordings':len(unmatched),
                'admission':'pending','note':'headers are not full decode; failed sample/resource limits require review'}
        _write(run/'inspection_summary.json',result);_write(run/'run_state.json',{'run_id':run_id,'status':result['status'],'updated_utc':_now()})
        return {'output':str(run),**result}
    finally:
        if db is not None: db.close()
        lock.unlink(missing_ok=True)


def reconcile(root, config, snapshot, pairs, inspection):
    root=Path(root).resolve();snapshot=Path(snapshot).resolve();pairs=Path(pairs).resolve();inspection=Path(inspection).resolve()
    m,q=_check_inputs(root,snapshot,pairs)
    boundary=(root/'data/manifests/phase2_inspection').resolve()
    if not inspection.is_relative_to(boundary) or inspection==boundary: raise ValueError('inspection outside expected root')
    ins=json.loads((inspection/'inspection_summary.json').read_text(encoding='utf8'))
    if ins['pairing_summary_sha256']!=_digest(pairs/'pairing_summary.json'): raise ValueError('inspection uses different pairing run')
    for name,digest in ins['artifacts_sha256'].items():
        if Path(name).name!=name or _digest(inspection/name)!=digest: raise ValueError('inspection artifact hash mismatch: '+name)
    prov=json.loads((snapshot/'provenance.json').read_text(encoding='utf8'))
    inventory=(root/prov['input_inventory']).resolve()
    if not inventory.is_relative_to((root/config['outputs']['inventory']).resolve()): raise ValueError('inventory outside expected root')
    inventory_summary=json.loads((inventory/'scan_summary.json').read_text(encoding='utf8'))
    if _digest(inventory/'scan_summary.json')!=prov['inventory_summary_sha256']: raise ValueError('inventory summary changed')
    for name in ('file_inventory.jsonl','transcript_inventory.jsonl'):
        if _digest(inventory/name)!=inventory_summary['artifacts_sha256'][name]: raise ValueError('inventory hash mismatch: '+name)
    raw=Counter();byte_counts=Counter()
    with (inventory/'file_inventory.jsonl').open(encoding='utf8') as inp:
        for line in inp:
            row=json.loads(line); raw[row['component']]+=1;byte_counts[row['component']]+=row['size_bytes'] or 0
    map_counts=Counter()
    with (snapshot/'mapping_evidence.jsonl').open(encoding='utf8') as inp:
        for line in inp:
            row=json.loads(line);map_counts[row['component']]+=1
    sample=Counter()
    with (inspection/'sample_decodes.jsonl').open(encoding='utf8') as inp:
        for line in inp:
            row=json.loads(line);sample[(row['component'],row['state'])]+=1
    components={}
    for comp in COMPONENTS:
        head=ins['component_header_counts'].get(comp,{})
        components[comp]={'files_hashed_inventory':raw[comp],'bytes_inventory':byte_counts[comp],
             'candidate_audio_records':map_counts[comp], 'candidate_class':'genuine' if comp=='genuine' else 'tts',
             'candidate_generator':None if comp=='genuine' else comp,'language':'unknown',
             'headers':head,'sample_decodes':{state:count for (c,state),count in sample.items() if c==comp},
             'admission':'pending'}
    issues=[]
    for comp in COMPONENTS:
        if raw[comp]!=map_counts[comp]: issues.append(f'{comp}: inventory and candidate audio counts differ')
        if sum(ins['component_header_counts'].get(comp,{}).values())!=map_counts[comp]: issues.append(f'{comp}: header count differs')
    if raw['transcripts']!=q['transcript_diagnostics'].get('ok',0): issues.append('transcript count differs from P2-06 successful checks')
    if q['audio_dispositions']!=m['counts']['recordings']: issues.append('audio pairing dispositions differ from recordings')
    if ins['status']!='ready_for_review': issues.append('P2-07 inspection requires attention')
    stages={'publisher':{'genuine':28539,'all_synthetic_including_vc':199773,'scope':'publisher release 1.0.1; synthetic includes TTS and VC, not directly comparable'},
            'downloaded':{'state':'user_reported_upload','independent_archive_comparison':'not_run'},
            'hashed_inventory':{'files':dict(raw),'bytes':dict(byte_counts),'note':'inventory updated after reported duplicate cleanup; retained hashes predate cleanup'},
            'header_inspected':{'counts':ins['component_header_counts']},'sample_decoded':{'counts':ins['sample_outcomes'],'scope':'representative sample only'},
            'in_scope':{'state':'pending_admission'},'deduplicated':{'state':'not_run'},'partitioned':{'state':'not_run'},'sampled_pilot':{'state':'not_run'}}
    outbase=(root/'reports/phase2').resolve()
    if not outbase.is_relative_to(root): raise ValueError('unsafe report root')
    outbase.mkdir(parents=True,exist_ok=True)
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:12]
    out=outbase/('p2_08-'+run_id);out.mkdir(exist_ok=False)
    result={'task':'P2-08','run_id':run_id,'status':'needs_attention' if issues else 'ready_for_review',
            'components':components,'transcript_files':raw['transcripts'],'unique_genuine_controls':raw['genuine'],
            'pairing':{'candidate_edges':q['candidate_edges'],'verified_edges':q['verified_edges'],
                       'identity':'user reports review; machine-readable P2-06 evidence remains pending'},
            'stages':stages,'issues':issues,'inspection_status':ins['status'],'source_admission':'pending',
            'inputs':{'snapshot_manifest_sha256':_digest(snapshot/'snapshot_manifest.json'),
                      'pairing_summary_sha256':_digest(pairs/'pairing_summary.json'),
                      'inspection_summary_sha256':_digest(inspection/'inspection_summary.json')}}
    _write(out/'reconciliation.json',result)
    card=['# PhonemeDF source card','',f'Generated UTC: {_now()}','',
          'Source: Zenodo release 1.0.1, user-attested; publisher license CC BY 4.0, subject to applicable underlying conditions.',
          'Local archive checksums have not been independently matched. Language and voice identities require evidence.',
          'User reports relationship and identity review; machine-readable P2-06 links remain candidates.',
          'PhonemeDF genuine and four TTS components are in scope; voice conversion is excluded.',
          'ChatterboxTTS is protected as the proposed final unseen family.','',
          '| Component | Inventory files | Candidate audio records | Header outcomes |','|---|---:|---:|---|']
    for comp in COMPONENTS:
        c=components[comp];card.append(f'| {comp} | {c["files_hashed_inventory"]} | {c["candidate_audio_records"]} | {_json(c["headers"])} |')
    card += ['',f'Transcript files: {raw["transcripts"]}. Unique genuine controls are counted once: {raw["genuine"]}.',
             f'P2-07 sample outcomes: {_json(ins["sample_outcomes"])}. Sample results do not prove all files decodable.',
             'In-scope admission, full deduplication, partitioning and pilot selection are not complete.',
             'Retained file hashes predate the user-reported duplicate cleanup; no post-cleanup full audio rehash has been performed.',
             f'Reconciliation issues: {_json(issues)}.']
    (out/'phonemedf.md').write_text('\n'.join(card)+'\n',encoding='utf8')
    _write(out/'artifact_manifest.json',{'reconciliation.json':_digest(out/'reconciliation.json'),'phonemedf.md':_digest(out/'phonemedf.md')})
    return {'output':str(out),**result}


def capacity(root, *, drive_free_gb=None, drive_total_gb=None, drive_used_gb=None, scratch_path=None, remaining_hours=None, planned_download_bytes=0,
             expanded_bytes=0, staging_bytes=0, cache_bytes=0):
    root=Path(root).resolve()
    for key,value in [('drive_free_gb',drive_free_gb),('drive_total_gb',drive_total_gb),('drive_used_gb',drive_used_gb),('remaining_hours',remaining_hours)]:
        if value is not None and (type(value) not in (int,float) or not math.isfinite(value) or value<0): raise ValueError(key+' must be a nonnegative finite number or null')
    for key,value in [('planned_download_bytes',planned_download_bytes),('expanded_bytes',expanded_bytes),('staging_bytes',staging_bytes),('cache_bytes',cache_bytes)]:
        if type(value) is not int or value<0: raise ValueError(key+' must be a nonnegative integer')
    if planned_download_bytes and not expanded_bytes: raise ValueError('planned download requires an expanded size estimate')
    scratch=Path(scratch_path or root).resolve();usage=shutil.disk_usage(scratch)
    subtotal=sum((planned_download_bytes,expanded_bytes,staging_bytes,cache_bytes));required=math.ceil(subtotal*1.25)
    drive_bytes=None if drive_free_gb is None else int(drive_free_gb*1_000_000_000)
    result={'task':'P2-09','captured_utc':_now(),'drive_free_gb_user_supplied':drive_free_gb,
        'drive_total_gb_user_supplied':drive_total_gb,'drive_used_gb_user_supplied':drive_used_gb,'drive_units':'decimal GB',
        'scratch_path':str(scratch),'scratch_free_bytes':usage.free,'remaining_hours_user_supplied':remaining_hours,
        'planned_download_bytes':planned_download_bytes,'expanded_bytes':expanded_bytes,'staging_bytes':staging_bytes,'cache_bytes':cache_bytes,
        'reserve_fraction':0.25,'required_additional_bytes_with_reserve':required,
        'drive_capacity_check':'not_required_no_acquisition' if required==0 else ('pending' if drive_bytes is None else ('pass' if drive_bytes>=required else 'blocked')),
        'scratch_capacity_check':'not_required_no_acquisition' if required==0 else ('pass' if usage.free>=required else 'blocked'),
        'work_time':'pending' if remaining_hours is None else 'provided',
        'acquisition':'none_planned' if planned_download_bytes==0 else 'proposed_only',
        'note':'Drive account quota must be provided separately; scratch free space is not Drive quota. Existing originals are not counted twice.'}
    result['status']='needs_attention' if ('blocked' in (result['drive_capacity_check'],result['scratch_capacity_check']) or result['drive_capacity_check']=='pending' or remaining_hours is None or any(x is None for x in (drive_free_gb,drive_total_gb,drive_used_gb))) else 'ready_for_review'
    base=root/'reports/phase2';base.mkdir(parents=True,exist_ok=True)
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:12]
    out=base/('p2_09-'+run_id);out.mkdir(exist_ok=False)
    _write(out/'acquisition_plan.json',result)
    (out/'capacity_worksheet.md').write_text('# P2-09 capacity worksheet\n\n```json\n'+json.dumps(result,indent=2)+'\n```\n',encoding='utf8')
    _write(out/'artifact_manifest.json',{'acquisition_plan.json':_digest(out/'acquisition_plan.json'),'capacity_worksheet.md':_digest(out/'capacity_worksheet.md')})
    return {'output':str(out),**result}
