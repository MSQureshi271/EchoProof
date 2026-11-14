"""Peak process memory (Windows peak working set) for one bundle: load, then one analyze() on a 10 s and a 60 s clip.
Run one model per fresh process. Usage: python scripts/r3_all/gr3_memory.py --model {lr,bigru,wavlm} --bundle PATH --out FILE.json"""
import argparse, ctypes, json, sys, time
from ctypes import wintypes
from pathlib import Path
sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts')
import r3_n2_latency_local as H


class PMC(ctypes.Structure):
    _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD), ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t),
                ('QuotaPeakPagedPoolUsage', ctypes.c_size_t), ('QuotaPagedPoolUsage', ctypes.c_size_t), ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
                ('QuotaNonPagedPoolUsage', ctypes.c_size_t), ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t)]


def mem():
    p = PMC(); p.cb = ctypes.sizeof(p)
    k, ps = ctypes.WinDLL('kernel32'), ctypes.WinDLL('psapi')
    k.GetCurrentProcess.restype = wintypes.HANDLE
    ps.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
    ps.GetProcessMemoryInfo.restype = wintypes.BOOL
    if not ps.GetProcessMemoryInfo(k.GetCurrentProcess(), ctypes.byref(p), p.cb):
        raise OSError('GetProcessMemoryInfo failed')
    return {'peak_working_set_mb': round(p.PeakWorkingSetSize / 2**20, 1), 'working_set_mb': round(p.WorkingSetSize / 2**20, 1), 'peak_pagefile_mb': round(p.PeakPagefileUsage / 2**20, 1)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--model', required=True); ap.add_argument('--bundle', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args()
    if Path(a.out).exists(): raise SystemExit('refusing to overwrite')
    root = Path('.').resolve(); res = {'model': a.model, 'bundle': a.bundle, 'after_import': mem()}
    import echoproof.inference as inf
    loaded = H.load_bundle(a.model, Path(a.bundle)); res['after_load'] = mem()
    for s in (10, 60):
        audio = H.clip_bytes(root, s); r = inf.analyze(audio, loaded); res[f'after_{s}s_clip'] = mem(); res[f'score_{s}s'] = float(r.result.score) if hasattr(r, 'result') else None
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2), encoding='utf-8', newline='\n'); print(json.dumps(res))


main()
