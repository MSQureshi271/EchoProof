"""Resumable, streaming file inventory for the Phase 2 primary dataset."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sqlite3
import stat
import time
import unicodedata
import uuid
from datetime import datetime, timezone


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _digest(path, block=1024 * 1024):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for part in iter(lambda: stream.read(block), b''):
            h.update(part)
    return h.hexdigest()


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _atomic(path, data):
    temp = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    with open(temp, 'x', encoding='utf-8', newline='\n') as out:
        out.write(data); out.flush(); os.fsync(out.fileno())
    os.replace(temp, path)


def _walk(root):
    """Yield files and explicit traversal failures; never follow symbolic links."""
    def visit(folder, relative):
        try:
            with os.scandir(folder) as entries:
                names = sorted((entry.name for entry in entries))
        except OSError as exc:
            yield relative or '.', 'error', f'{type(exc).__name__}: {exc}'
            return
        for name in names:
            rel = f'{relative}/{name}' if relative else name
            p = folder / name
            try:
                mode = p.lstat().st_mode
                if stat.S_ISLNK(mode):
                    yield rel, 'error', 'symbolic link not followed'
                elif stat.S_ISDIR(mode):
                    yield from visit(p, rel)
                elif stat.S_ISREG(mode):
                    yield rel, 'file', None
                else:
                    yield rel, 'error', 'unsupported file type'
            except OSError as exc:
                yield rel, 'error', f'{type(exc).__name__}: {exc}'
    yield from visit(root, '')


def _hash_stable(path, block, retries):
    for attempt in range(retries + 1):
        try:
            before = path.lstat()
            if not stat.S_ISREG(before.st_mode):
                return None, None, None, 'error', 'not a regular file'
            flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0)
            fd = os.open(path, flags)
            h = hashlib.sha256()
            with os.fdopen(fd, 'rb') as stream:
                opened = os.fstat(stream.fileno())
                while True:
                    part = stream.read(block)
                    if not part: break
                    h.update(part)
                after_fd = os.fstat(stream.fileno())
            after = path.lstat()
            signature = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
            if signature(before) == signature(opened) == signature(after_fd) == signature(after):
                return h.hexdigest(), before.st_size, before.st_mtime_ns, 'completed', None
            failure = 'file changed during hashing'
        except OSError as exc:
            failure = f'{type(exc).__name__}: {exc}'
            if attempt == retries:
                return None, None, None, 'error', failure
    return None, None, None, 'unstable', failure


def _id(component, relative):
    return 'phonemedf:' + hashlib.sha256((component + '\0' + relative).encode('utf-8')).hexdigest()


def _write_jsonl(path, rows):
    temp = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    h = hashlib.sha256()
    with open(temp, 'xb') as out:
        for row in rows:
            data = (_json(row) + '\n').encode('utf-8')
            out.write(data); h.update(data)
        out.flush(); os.fsync(out.fileno())
    os.replace(temp, path)
    return h.hexdigest()


def run_inventory(root, config, config_path, registry_path, resume=None, *, stop_after=None):
    """Scan PhonemeDF only; stop_after is reserved for deterministic recovery tests."""
    root = Path(root).resolve()
    config_path, registry_path = Path(config_path), Path(registry_path)
    outbase = (root / config['outputs']['inventory']).resolve()
    outbase.mkdir(parents=True, exist_ok=True)
    run_id = resume or datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:12]
    if not run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-' for c in run_id):
        raise ValueError('invalid run id')
    run = outbase / run_id
    if resume:
        if not run.is_dir() or not (run / 'checkpoint.sqlite').is_file(): raise ValueError('resume checkpoint not found')
    else:
        run.mkdir(exist_ok=False)
    print(f'Inventory run ID: {run_id}', flush=True)
    fingerprint = {'schema_version': 1, 'source': 'phonemedf', 'config_sha256': _digest(config_path),
                   'registry_sha256': _digest(registry_path), 'project_root': str(root),
                   'implementation_sha256': _digest(Path(__file__)),
                   'config_loader_sha256': _digest(Path(__file__).parent.parent / 'phase2_config.py')}
    lock = run / 'running.lock'
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError('run is locked; verify prior process stopped before removing running.lock') from exc
    os.close(fd)
    db = None
    try:
        db = sqlite3.connect(run / 'checkpoint.sqlite')
        db.execute('PRAGMA journal_mode=DELETE')
        db.execute('PRAGMA synchronous=FULL')
        db.execute('CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
        db.execute('CREATE TABLE IF NOT EXISTS rows (component TEXT NOT NULL, path TEXT NOT NULL, row_json TEXT NOT NULL, row_hash TEXT NOT NULL, seen INTEGER NOT NULL, PRIMARY KEY(component,path))')
        if db.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
            raise ValueError('checkpoint database integrity check failed')
        prior = db.execute("SELECT value FROM meta WHERE key='fingerprint'").fetchone()
        if prior and json.loads(prior[0]) != fingerprint: raise ValueError('resume config, registry, source location or implementation changed')
        if resume and not prior: raise ValueError('missing checkpoint fingerprint')
        if not prior:
            db.execute("INSERT INTO meta VALUES ('fingerprint',?)", (_json(fingerprint),)); db.commit()
        for value, expected in db.execute('SELECT row_json,row_hash FROM rows'):
            if hashlib.sha256(value.encode('utf-8')).hexdigest() != expected: raise ValueError('checkpoint row checksum mismatch')
        previous_summary = run / 'scan_summary.json'
        if previous_summary.exists():
            os.replace(previous_summary, run / ('scan_summary.previous.' + uuid.uuid4().hex + '.json'))
        _atomic(run/'run_state.json', _json({'run_id':run_id, 'status':'running', 'started_utc':_utc()})+'\n')
        generation = int(time.time_ns())
        started = time.monotonic(); total = 0; batch = 0
        for component in ('genuine', 'transcripts', 'xtts', 'chatterbox', 'melo', 'vits'):
            source = root / config['sources'][component]
            for rel, kind, walk_error in _walk(source):
                old = db.execute('SELECT row_json FROM rows WHERE component=? AND path=?', (component, rel)).fetchone()
                cached = json.loads(old[0]) if old else None
                if cached and config['immutable_inputs'] and cached['state'] == 'completed' and kind == 'file':
                    row = cached
                else:
                    begin = _utc()
                    sha, size, mtime, state, failure = _hash_stable(source / rel, config['block_bytes'], config['retries']) if kind == 'file' else (None,None,None,'error',walk_error)
                    row = {'schema_version':1, 'scope':'research', 'file_id':_id(component,rel), 'source':'phonemedf',
                           'component':component, 'relative_path':rel, 'file_type':Path(rel).suffix.lower() or 'unknown',
                           'size_bytes':size, 'scan_started_utc':begin, 'scan_finished_utc':_utc(),
                           'mtime_ns':mtime, 'sha256':sha, 'failure':failure, 'state':state}
                raw = _json(row)
                db.execute('INSERT OR REPLACE INTO rows VALUES (?,?,?,?,?)', (component,rel,raw,hashlib.sha256(raw.encode('utf-8')).hexdigest(),generation))
                total += 1; batch += 1
                if batch >= config['checkpoint_interval']:
                    db.commit(); batch = 0
                    print(f'Inventory checkpoint: {total} paths processed', flush=True)
                if stop_after is not None and total >= stop_after:
                    db.commit()
                    return {'run_id':run_id,'status':'interrupted_fixture','processed':total}
        db.execute('DELETE FROM rows WHERE seen != ?', (generation,)); db.commit()
        records = lambda where='1': (json.loads(x[0]) for x in db.execute(f'SELECT row_json FROM rows WHERE {where} ORDER BY component,path'))
        hashes = {}
        hashes['file_inventory.jsonl'] = _write_jsonl(run/'file_inventory.jsonl', records())
        hashes['transcript_inventory.jsonl'] = _write_jsonl(run/'transcript_inventory.jsonl', records("component='transcripts'"))
        hashes['hash_ledger.jsonl'] = _write_jsonl(run/'hash_ledger.jsonl', ({k:r[k] for k in ('file_id','component','relative_path','sha256','state','failure')} for r in records()))
        content = hashlib.sha256()
        counts = {'completed':0,'error':0,'unstable':0,'pending':0}; byte_count = 0
        collisions = {}
        for row in records():
            counts[row['state']] += 1; byte_count += row['size_bytes'] or 0
            key = (row['component'],unicodedata.normalize('NFC',row['relative_path']).casefold())
            collisions.setdefault(key,[]).append(row['relative_path'])
            stable = {k:row[k] for k in ('file_id','component','relative_path','file_type','size_bytes','sha256','state','failure')}
            content.update((_json(stable)+'\n').encode('utf-8'))
        collision_count = sum(1 for v in collisions.values() if len(v)>1)
        collision_rows = ({'component':k[0],'normalized_path':k[1],'paths':v} for k,v in sorted(collisions.items()) if len(v)>1)
        hashes['collisions.jsonl'] = _write_jsonl(run/'collisions.jsonl', collision_rows)
        summary = {'schema_version':1,'run_id':run_id,'status':'complete' if not (counts['error'] or counts['unstable'] or counts['pending'] or collision_count) else 'needs_attention',
                   'counts':counts,'bytes_hashed_or_reused':byte_count,'elapsed_seconds':round(time.monotonic()-started,3),
                   'collision_groups':collision_count,
                   'content_digest_sha256':content.hexdigest(),'artifacts_sha256':hashes,'fingerprint':fingerprint,
                   'hash_reuse':'immutable_inputs' if config['immutable_inputs'] else 'full_rehash_on_resume',
                   'scope':'inventory only; no audio decoding or dataset admission'}
        _atomic(run/'scan_summary.json',_json(summary)+'\n')
        _atomic(run/'run_state.json',_json({'run_id':run_id,'status':summary['status'],'finished_utc':_utc()})+'\n')
        return summary
    finally:
        if db is not None: db.close()
        lock.unlink(missing_ok=True)
