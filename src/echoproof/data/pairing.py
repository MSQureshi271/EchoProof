"""P2-06 candidate transcript and audio relationships for PhonemeDF."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import unicodedata
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from ..contracts import Recording
from .inventory import _digest, _json


def _normal(text):
    return ' '.join(unicodedata.normalize('NFKC',text).split())


def _transcript(path, expected):
    """Read bounded transcript bytes and compare against inventory hash."""
    try:
        with path.open('rb') as f:
            raw = f.read(1048577)
        if len(raw)>1048576: return {'state':'resource_limit','raw_sha256':None,'normalized_sha256':None}
        digest=hashlib.sha256(raw).hexdigest()
        if digest != expected: return {'state':'hash_mismatch','raw_sha256':digest,'normalized_sha256':None}
        try: text=raw.decode('utf-8-sig')
        except UnicodeDecodeError: return {'state':'invalid_utf8','raw_sha256':digest,'normalized_sha256':None}
        normalized=_normal(text)
        return {'state':'ok' if normalized else 'empty', 'raw_sha256':digest,
                'normalized_sha256':hashlib.sha256(normalized.encode('utf8')).hexdigest() if normalized else None,
                'utf8_bom':raw.startswith(b'\xef\xbb\xbf')}
    except OSError as exc:
        return {'state':'read_error','raw_sha256':None,'normalized_sha256':None,'error':str(exc)}


def run_pairing(root, config, snapshot):
    import pyarrow.parquet as pq
    root=Path(root).resolve(); snapshot=Path(snapshot).resolve()
    boundary=(root/'data/manifests/phase2_raw').resolve()
    if not snapshot.is_relative_to(boundary) or snapshot==boundary: raise ValueError('snapshot outside raw snapshot root')
    manifest=json.loads((snapshot/'snapshot_manifest.json').read_text(encoding='utf8'))
    if manifest.get('status')!='ready_for_review': raise ValueError('snapshot not ready for review')
    for name,expected in manifest['artifacts_sha256'].items():
        if Path(name).name!=name or _digest(snapshot/name)!=expected: raise ValueError('snapshot artifact mismatch: '+name)
    provenance=json.loads((snapshot/'provenance.json').read_text(encoding='utf8'))
    inventory=(root/provenance['input_inventory']).resolve()
    invroot=(root/config['outputs']['inventory']).resolve()
    if not inventory.is_relative_to(invroot): raise ValueError('inventory path escapes boundary')
    summary=inventory/'scan_summary.json'
    if _digest(summary)!=provenance['inventory_summary_sha256']: raise ValueError('inventory summary changed')
    meta=json.loads(summary.read_text(encoding='utf8'))
    trans=inventory/'transcript_inventory.jsonl'
    if _digest(trans)!=meta['artifacts_sha256']['transcript_inventory.jsonl']: raise ValueError('transcript inventory hash mismatch')
    components={str(PurePosixPath(config['sources'][k])):k for k in ('genuine','xtts','chatterbox','melo','vits')}
    groups=defaultdict(lambda:defaultdict(list)); seen=set(); audio_count=0
    parquet=pq.ParquetFile(snapshot/'raw_recordings.parquet')
    if parquet.schema_arrow.names != ['schema_name','schema_version','payload_json']: raise ValueError('unexpected recording envelope')
    for batch in parquet.iter_batches(batch_size=2048):
        for row in batch.to_pylist():
            if row['schema_name']!='recording' or row['schema_version']!=1: raise ValueError('invalid recording envelope')
            record=Recording.from_json(row['payload_json'])
            if record.recording_id in seen: raise ValueError('duplicate recording ID')
            seen.add(record.recording_id); audio_count+=1
            path=PurePosixPath(record.original_path)
            matching=[(len(PurePosixPath(prefix).parts),key,path.relative_to(PurePosixPath(prefix))) for prefix,key in components.items() if path.is_relative_to(PurePosixPath(prefix))]
            if len(matching)!=1: raise ValueError('recording path does not match one component')
            _,component,relative=matching[0]
            groups[relative.with_suffix('').as_posix()][component].append(record.recording_id)
    if audio_count!=manifest['counts']['recordings']: raise ValueError('recording count mismatch')
    transcript_rows=[]
    transcript_root=(root/config['sources']['transcripts']).resolve()
    with trans.open(encoding='utf8') as inp:
        for line in inp:
            row=json.loads(line)
            rel=PurePosixPath(row['relative_path'])
            if rel.is_absolute() or '\\' in row['relative_path'] or any(x in ('','.','..') for x in row['relative_path'].split('/')): raise ValueError('unsafe transcript path')
            path=(transcript_root/Path(*rel.parts)).resolve()
            if not path.is_relative_to(transcript_root): raise ValueError('transcript path escapes root')
            diagnostic=_transcript(path,row.get('sha256')) if row.get('state')=='completed' and row.get('file_type')=='.txt' else {'state':'invalid_inventory_entry','raw_sha256':None,'normalized_sha256':None}
            diagnostic.update({'file_id':row['file_id'],'relative_path':row['relative_path']})
            transcript_rows.append(diagnostic)
            groups[rel.with_suffix('').as_posix()]['transcripts'].append(row['file_id'])
    if audio_count + len(transcript_rows) != manifest['counts']['rows_seen']:
        raise ValueError('input rows include unaccounted dispositions; review before pairing')
    output=(root/'data/manifests/phase2_pairs').resolve()
    if not output.is_relative_to(root): raise ValueError('output escapes project root')
    output.mkdir(parents=True,exist_ok=True)
    run_id=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:12]
    out=output/run_id; out.mkdir(exist_ok=False)
    counts=Counter(); diagnostics=Counter(); edge_count=0; audio_dispositions=0
    with (out/'candidate_edges.jsonl').open('x',encoding='utf8',newline='\n') as edges, (out/'pair_dispositions.jsonl').open('x',encoding='utf8',newline='\n') as dispositions, (out/'coverage.jsonl').open('x',encoding='utf8',newline='\n') as coverage, (out/'transcript_diagnostics.jsonl').open('x',encoding='utf8',newline='\n') as details:
        for d in transcript_rows: details.write(_json(d)+'\n');diagnostics[d['state']]+=1
        for stem,by_component in sorted(groups.items()):
            present={key:len(by_component.get(key,[])) for key in ('genuine','xtts','chatterbox','melo','vits','transcripts')}
            counts['groups']+=1
            if all(value==1 for value in present.values()): counts['complete_six_way_unique']+=1
            if any(value>1 for value in present.values()): counts['groups_with_multiplicity']+=1
            if not present['genuine']: counts['groups_without_genuine']+=1
            if not present['transcripts']: counts['groups_without_transcript']+=1
            coverage.write(_json({'candidate_stem':stem,'counts':present})+'\n')
            audio=[(key,item) for key in ('genuine','xtts','chatterbox','melo','vits') for item in by_component.get(key,[])]
            # Star edges retain candidate linkage even for groups missing genuine audio.
            endpoints=[item for _,item in audio]+by_component.get('transcripts',[])
            for endpoint in endpoints[1:]:
                edges.write(_json({'endpoint_a':endpoints[0],'endpoint_b':endpoint,'relation_type':'candidate_same_stem','evidence':'shared source-relative stem','state':'candidate'})+'\n');edge_count+=1
            for component,item in audio:
                dispositions.write(_json({'recording_id':item,'component':component,'candidate_stem':stem,'state':'pending','reason':'identity_and_pairing_unverified','has_genuine':bool(present['genuine']),'has_transcript':bool(present['transcripts'])})+'\n');audio_dispositions+=1
        for f in (edges,dispositions,coverage,details): f.flush();os.fsync(f.fileno())
    (out/'verified_edges.jsonl').write_text('',encoding='utf8')
    identity={'source_utterance_id':'unknown','source_speaker_id':'unknown','output_voice_id':'unknown','reference_id':'unknown','verification':'pending','normalization':'NFKC + Unicode whitespace collapse, v1; UTF-8 encoded hash', 'manual_review_coverage':0}
    (out/'identity_coverage.json').write_text(_json(identity)+'\n',encoding='utf8')
    names=['candidate_edges.jsonl','verified_edges.jsonl','pair_dispositions.jsonl','coverage.jsonl','transcript_diagnostics.jsonl','identity_coverage.json']
    result={'task':'P2-06','run_id':run_id,'status':'ready_for_review' if diagnostics and all(k=='ok' for k in diagnostics) else 'needs_attention',
            'snapshot_manifest_sha256':_digest(snapshot/'snapshot_manifest.json'),'snapshot':str(snapshot.relative_to(root)),
            'counts':dict(counts),'audio_recordings':audio_count,'audio_dispositions':audio_dispositions,'candidate_edges':edge_count,
            'transcript_diagnostics':dict(diagnostics),'verified_edges':0,'artifacts_sha256':{n:_digest(out/n) for n in names},
            'scope':'filename relationships are candidates; no audio identity verified and no split assigned'}
    (out/'pairing_summary.json').write_text(_json(result)+'\n',encoding='utf8')
    return {'output':str(out),**result}
