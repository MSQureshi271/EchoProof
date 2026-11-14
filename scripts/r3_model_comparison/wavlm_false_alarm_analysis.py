import sys, json, collections
import numpy as np, pyarrow.parquet as pq
R='runs/r3/r3_n1_wavlm/20261114T083533Z-e6ff5a288068'
thr=0.4040371283027592
P=pq.read_table(R+'/operating/operating_predictions.parquet').to_pylist()
C={r['recording_id']:r for r in pq.read_table('data/manifests/cohort_v1.parquet').to_pylist()}
for p in P: p.update(dur=C[p['recording_id']]['duration_s'],chap=C[p['recording_id']]['chapter_id'],gen=C[p['recording_id']]['generator'],qf=C[p['recording_id']]['quality_flags'],spk=C[p['recording_id']]['speaker_id'])
g=[p for p in P if p['label']==0]; t=[p for p in P if p['label']==1]
fp=[p for p in g if p['decision']==1]; tn=[p for p in g if p['decision']==0]
print('genuine',len(g),'FP',len(fp),'TTS',len(t))
q=lambda a,x:np.percentile([p[x] for p in a],[10,25,50,75,90]).round(2)
print('dur FP',q(fp,'dur'),'TN',q(tn,'dur'),'TTS',q(t,'dur'))
print('flags',collections.Counter(p['qf'] for p in g).most_common(5),'FP flags',collections.Counter(p['qf'] for p in fp).most_common(5))
print('\nFP rate by duration bin (genuine):')
for lo,hi in [(0,4),(4,6),(6,8),(8,10),(10,13),(13,16),(16,40)]:
    s=[p for p in g if lo<=p['dur']<hi]; 
    if s: print(f'{lo}-{hi}s n={len(s)} FP={sum(p["decision"] for p in s)} rate={np.mean([p["decision"] for p in s]):.3f}  TTS-in-bin n={sum(1 for p in t if lo<=p["dur"]<hi)} recall={np.mean([p["decision"] for p in t if lo<=p["dur"]<hi]) if any(lo<=p["dur"]<hi for p in t) else float("nan"):.3f}')
print('\nper speaker (genuine): n, FP, dur median, median score, max score')
bys=collections.defaultdict(list)
for p in g: bys[p['spk']].append(p)
for s,v in sorted(bys.items(),key=lambda kv:-sum(x['decision'] for x in kv[1])):
    print(s,len(v),sum(x['decision'] for x in v),round(np.median([x['dur'] for x in v]),1),round(np.median([x['deployed_score'] for x in v]),3),round(max(x['deployed_score'] for x in v),3),'chapters',len({x['chap'] for x in v}))
print('\nFP concentration by chapter:',collections.Counter(p['chap'] for p in fp).most_common(8))
# score distribution of genuine & margin
sc=np.array([p['deployed_score'] for p in g]);print('genuine score pct 50/75/90/95/99',np.percentile(sc,[50,75,90,95,99]).round(3))
band=lambda a,b:sum(1 for p in g if a<=p['deployed_score']<b)
print('genuine in [0.404,0.6)',band(.404,.6),'[0.6,0.9)',band(.6,.9),'>=0.9',band(.9,2))
print('FP score pct',np.percentile([p['deployed_score'] for p in fp],[10,50,90]).round(3))
tt=np.array([p['deployed_score'] for p in t]);print('TTS missed (<thr):',int((tt<thr).sum()),'; TTS in [0.404,0.6):',int(((tt>=thr)&(tt<.6)).sum()))
# TTS by generator
print('\nTTS by generator:')
bg=collections.defaultdict(list)
for p in t: bg[p['gen']].append(p)
for k,v in bg.items(): print(k,len(v),'recall',round(np.mean([x['decision'] for x in v]),3),'dur med',round(np.median([x['dur'] for x in v]),1))
# abstention band simulation (operating, descriptive)
print('\nabstention bands on operating (descriptive): band lo..thr..hi -> coverage, FP, FPR among accepted, recall among accepted')
for lo,hi in [(0.404,0.404),(0.2,0.7),(0.1,0.8),(0.05,0.9),(0.02,0.95)]:
    acc=[p for p in P if not (lo<=p['deployed_score']<hi)] if hi>lo else P
    ag=[p for p in acc if p['label']==0]; at=[p for p in acc if p['label']==1]
    print(lo,hi,'coverage',round(len(acc)/len(P),3),'FP',sum(p['decision'] for p in ag),'FPR',round(np.mean([p['decision'] for p in ag]),3),'recall',round(np.mean([p['decision'] for p in at]),3))
# correlation duration/score within genuine
print('\nspearman-ish dur vs score genuine:',np.corrcoef(np.argsort(np.argsort([p['dur'] for p in g])),np.argsort(np.argsort(sc)))[0,1].round(3))
