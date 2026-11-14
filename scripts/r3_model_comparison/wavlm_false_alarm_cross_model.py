import numpy as np, pyarrow.parquet as pq, collections
def L(p,col='raw_score'): return {r['recording_id']:r for r in pq.read_table(p).to_pylist()}
W=L('runs/r3/r3_n1_wavlm/20261114T083533Z-e6ff5a288068/operating/operating_predictions.parquet')
M={'WavLM raw':({k:v['raw_score'] for k,v in W.items()}),
 'LR':{k:v['raw_score'] for k,v in L('runs/r3/r3_n1_lr/selected_lr_v1/export/predictions_recordings_operating.parquet').items()},
 'BiGRU':{k:v['raw_score'] for k,v in L('experiments/r3_m1/runs/batch01/selected/predictions_recordings_operating.parquet').items()},
 'AST':{k:v['raw_score'] for k,v in L('runs/r3/r3_u2_20261112T134049Z/predictions_recordings_operating.parquet').items()}}
for n,m in M.items(): assert set(m)==set(W),n
gen=[k for k,v in W.items() if v['label']==0]
fp=[k for k in gen if W[k]['decision']==1]; ok=[k for k in gen if W[k]['decision']==0]
spk=lambda k:W[k]['group_id']
def auc(pos,neg):
    p=np.array(pos);n=np.array(neg);return float(((p[:,None]>n[None,:]).mean()+0.5*(p[:,None]==n[None,:]).mean()))
print(f'{len(fp)} WavLM false alarms vs {len(ok)} other genuine\n')
print('model        | flagged@0.5 among the 49 | among other 211 | median score FP / other | AUC(FP vs other)')
for n,m in M.items():
    a=[m[k] for k in fp];b=[m[k] for k in ok]
    print(f'{n:12s} | {sum(x>=0.5 for x in a):2d}/49 ({np.mean([x>=0.5 for x in a]):.0%})        | {sum(x>=0.5 for x in b):3d}/211 ({np.mean([x>=0.5 for x in b]):.0%}) | {np.median(a):.3f} / {np.median(b):.4f} | {auc(a,b):.3f}')
# per-model all-genuine FP at 0.5 and overlap
print('\nGenuine flagged at raw 0.5 per model:',{n:sum(m[k]>=0.5 for k in gen) for n,m in M.items()})
others=['LR','BiGRU','AST']
flag={n:{k for k in gen if M[n][k]>=0.5} for n in M}
print('WavLM(raw 0.5) flagged:',len(flag['WavLM raw']))
for n in others: print(f'  overlap with {n}:',len(flag[n]&set(fp)),'of its',len(flag[n]),'flags within the 49')
allthree=set(fp)&flag['LR']&flag['BiGRU']&flag['AST'];print('flagged by WavLM and all 3 others:',len(allthree))
anyo=set(fp)&(flag['LR']|flag['BiGRU']|flag['AST']);print('flagged by WavLM and at least one other:',len(anyo))
print('flagged by no other model:',len(set(fp)-anyo))
# per speaker agreement
print('\nper speaker (genuine) : WavLM FP | LR | BiGRU | AST flags at 0.5 | mean score rank across 3 others')
by=collections.defaultdict(list)
for k in gen: by[spk(k)].append(k)
for s,v in sorted(by.items(),key=lambda kv:-sum(W[k]['decision'] for k in kv[1])):
    print(f"{s:14s} {sum(W[k]['decision'] for k in v):2d} | "+' | '.join(f"{sum(M[n][k]>=0.5 for k in v):2d}" for n in others)+f" | median score LR {np.median([M['LR'][k] for k in v]):.2f} BiGRU {np.median([M['BiGRU'][k] for k in v]):.3f} AST {np.median([M['AST'][k] for k in v]):.3f}")
# within the 4 problem speakers: do other models separate WavLM-FP from non-FP of same speaker?
print('\nwithin the 4 problem speakers, AUC of others separating the WavLM-flagged from the unflagged:')
P=[s for s in ['speaker:3699','speaker:6476','speaker:4406','speaker:5561']]
fa=[k for k in fp if spk(k) in P]; ua=[k for k in ok if spk(k) in P]
print(len(fa),len(ua),{n:round(auc([M[n][k] for k in fa],[M[n][k] for k in ua]),3) for n in M})
# rank correlation WavLM raw vs others across all genuine
def rk(x): return np.argsort(np.argsort(x))
for n in others: print('genuine spearman WavLM vs',n,round(float(np.corrcoef(rk([M['WavLM raw'][k] for k in gen]),rk([M[n][k] for k in gen]))[0,1]),3))
# TTS side: do the others miss what WavLM misses?
tts=[k for k,v in W.items() if v['label']==1]
miss={n:{k for k in tts if M[n][k]<0.5} for n in M}
print('\nTTS missed at raw 0.5:',{n:len(s) for n,s in miss.items()},'; WavLM deployed misses:',sum(W[k]['decision']==0 for k in tts))
