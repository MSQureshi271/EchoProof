import json, sys

d = json.load(open(sys.argv[1]))
OUT = sys.argv[2]
M = ['WavLM', 'AST', 'BiGRU', 'LR']
ROLES = ['selection', 'calibration', 'operating']
raw = d['raw_at_0_5']
p = lambda x, n=1: f'{100 * x:.{n}f}%'
f4 = lambda x: f'{x:.4f}'
f3 = lambda x: f'{x:.3f}'


def table(head, body):
    lines = ['| ' + ' | '.join(head) + ' |', '|' + '|'.join(['---'] + ['---:'] * (len(head) - 1)) + '|']
    lines += ['| ' + ' | '.join(r) + ' |' for r in body]
    return '\n'.join(lines)


def role_table(role):
    head = ['Model', 'Macro F1', 'ROC-AUC', 'EER', 'Avg precision', 'Genuine FPR', 'TTS recall', 'TTS precision', 'Accuracy']
    body = [[m] + [f4(raw[m][role][k]) for k in ('macro_f1', 'auc', 'eer', 'ap', 'fpr', 'recall', 'precision', 'acc')] for m in M]
    return table(head, body)


def gen_table():
    head = ['Model', 'Role', 'MeloTTS recall', 'VITS recall', 'XTTS recall', 'Genuine FPR']
    body = [[m, r, p(raw[m][r]['rec_melo']), p(raw[m][r]['rec_vits']), p(raw[m][r]['rec_xtts']), p(raw[m][r]['fpr'])] for r in ROLES for m in M]
    return table(head, body)


def prob_table():
    c = d['common_calibration_operating']
    head = ['Model', 'Log loss raw', 'Log loss temp.', 'Log loss affine', 'Brier raw', 'Brier temp.', 'Brier affine', 'ECE raw', 'ECE temp.', 'ECE affine']
    body = [[m] + [f4(c[m][k][x]) for x in ('logloss', 'brier', 'ece') for k in ('raw', 'temperature', 'affine')] for m in M]
    return table(head, body)


def own_table():
    o = d['teammate_calibrations_operating']
    return table(['Calibration as run by the owner', 'Log loss', 'Brier', 'ECE'], [[k, f4(v['logloss']), f4(v['brier']), f4(v['ece'])] for k, v in o.items()])


def params_table():
    c = d['calibration_params']
    return table(['Model', 'Temperature T', 'Affine a', 'Affine b'], [[m, f3(c[m]['T']), f3(c[m]['a']), f3(c[m]['b'])] for m in M])


def thr_table(src):
    t = d['thresholds_applied_to_operating']
    head = ['Model', 'Raw threshold', 'Macro F1', 'Genuine FPR', 'False alarms', 'TTS recall', 'Missed TTS', 'XTTS recall']
    body = [[m, f4(t[m][src]['threshold_raw']), f4(t[m][src]['macro_f1']), p(t[m][src]['fpr']), str(t[m][src]['fp']), p(t[m][src]['recall']), str(t[m][src]['fn']), p(t[m][src]['rec_xtts'])] for m in M]
    return table(head, body)


def ci_table():
    b = d['bootstrap']
    head = ['Model', 'ROC-AUC [95% CI]', 'EER [95% CI]', 'Macro F1 @0.5 [95% CI]', 'Genuine FPR @0.5 [95% CI]', 'TTS recall @0.5 [95% CI]']
    body = []
    for m in M:
        row = [m]
        for k in ('auc', 'eer', 'macro_f1', 'fpr', 'recall'):
            lo, hi = b['ci'][m][k]
            row.append(f"{b['point_raw_0_5'][m][k]:.3f} [{lo:.3f}, {hi:.3f}]")
        body.append(row)
    return table(head, body)


def paired_table():
    b = d['bootstrap']['paired_vs_wavlm']
    head = ['WavLM minus', 'ROC-AUC', 'EER', 'Macro F1 @0.5', 'Genuine FPR @0.5', 'TTS recall @0.5']
    body = []
    for m in ('AST', 'BiGRU', 'LR'):
        row = [m]
        for k in ('auc', 'eer', 'macro_f1', 'fpr', 'recall'):
            x = b[m][k]
            star = ' *' if x['ci'][0] > 0 or x['ci'][1] < 0 else ''
            row.append(f"{x['diff']:+.3f} [{x['ci'][0]:+.3f}, {x['ci'][1]:+.3f}]{star}")
        body.append(row)
    return table(head, body)


def conc_table():
    c = d['false_alarm_concentration_operating_threshold']
    head = ['Model', 'False alarms', 'Speakers with any', 'Speakers with at least 25% of their recordings flagged', 'Share held by the top 3 speakers', 'Worst speakers (flagged of 20)']
    body = [[m, str(c[m]['false_alarms']), f"{c[m]['speakers_with_any']} of 13", f"{c[m]['speakers_ge_25pct']} of 13", p(c[m]['top3_share'], 0),
             ', '.join(f"{k.split(':')[1]} ({a})" for k, a, n in c[m]['top'][:3])] for m in M]
    return table(head, body)


def dm_table():
    x = d['duration_matched_selection_raw_0_5']
    head = ['Model', 'Recordings', 'Macro F1', 'ROC-AUC', 'EER', 'Genuine FPR', 'TTS recall', 'XTTS recall']
    body = [[m, str(x[m]['n']), f4(x[m]['macro_f1']), f4(x[m]['auc']), f4(x[m]['eer']), p(x[m]['fpr']), p(x[m]['recall']), p(x[m]['rec_xtts'])] for m in M]
    return table(head, body)


def agree_table():
    a = d['agreement_operating']
    head = ['Pair', 'Same decision at raw 0.5', 'Rank correlation of scores']
    body = [[k.replace('~', ' and '), p(v['decision_agreement']), f3(v['rank_correlation'])] for k, v in a.items()]
    return table(head, body)


def ens_table():
    e = d['descriptive_unweighted_mean_ensembles_operating']
    head = ['Combination (unweighted mean of raw scores)', 'ROC-AUC', 'EER', 'Macro F1 @0.5', 'Genuine FPR', 'TTS recall']
    body = [[k, f4(v['auc']), f4(v['eer']), f4(v['macro_f1']), p(v['fpr']), p(v['recall'])] for k, v in e.items()]
    return table(head, body)


def rank_table():
    order = d['contract_ranking_selection_role']
    head = ['Rank', 'Model', 'Selection macro F1 @0.5', 'EER (tie-break 2)', 'Genuine FPR @0.5 (tie-break 3)']
    return table(head, [[str(i + 1), m, f4(raw[m]['selection']['macro_f1']), f4(raw[m]['selection']['eer']), p(raw[m]['selection']['fpr'])] for i, m in enumerate(order)])


hash_lines = '\n'.join(f'- `{k}`: sha256 `{v}…`' for k, v in d['hashes'].items())

TEXT = f"""# EchoProof Protocol B: model comparison (WavLM against LR, BiGRU and AST)

Generated 14 November 2026 from saved recording-level predictions. No model was run, no test partition was read, and no audio was used.

## 1. What this report is

It compares the four Protocol B candidates side by side on every metric the saved predictions support:

| Model | Selected configuration | Prediction source |
|---|---|---|
| WavLM-head-B | `mean_std` pooling, epoch 14 | `reports/r3/r3_u1_B/` |
| AST partial fine-tune | baseline recipe, epoch 1 | `runs/r3/r3_u2_20261112T134049Z/` |
| BiGRU | `b01_lower_lr_full20`, epoch 16 | `experiments/r3_m1/runs/batch01/selected/` |
| Logistic regression | `selected_lr_v1` | `runs/r3/r3_n1_lr/selected_lr_v1/export/` |

Validation roles (Protocol B, split hash `0854de6b…`, 75% TTS in every role):

| Role | Recordings | Speakers | Used for |
|---|---:|---:|---|
| selection | 2,000 | 25 | model and epoch selection, contract ranking |
| calibration | 960 | 12 | fitting calibration mappings |
| operating | 1,040 | 13 | evaluating calibration and choosing thresholds |

Every role contains unseen XTTS recordings beside MeloTTS, VITS and genuine speech. XTTS is the hardest generator for every model (LR also misses about 9% of VITS).

**How the comparison is kept fair.** The four models were calibrated by different people with different methods (LR by temperature scaling, BiGRU by Platt and isotonic scaling, AST not at all, WavLM by an affine mapping). Comparing those results directly would compare the calibration choices as much as the models. So the headline comparisons use **raw scores** and **one common reference calibration fitted the same way for every model on the calibration role only** (Section 7). The owners' own calibrations are listed separately.

**What it is not.** It is not the GR3 lock and it does not select a production model. It uses no test data and no Protocol A model.

## 2. Summary

1. **Under the contract's ranking rule WavLM ranks first** on the selection role (macro F1 {f4(raw['WavLM']['selection']['macro_f1'])}, EER {f4(raw['WavLM']['selection']['eer'])}, FPR {p(raw['WavLM']['selection']['fpr'])}), ahead of AST, BiGRU and LR. The margin over AST is visible in the selection-role point estimates (no interval was computed there). It does not carry over to the operating role.
2. **On the operating role WavLM, AST and BiGRU cannot be separated on ROC-AUC, EER, macro F1 or false-alarm rate.** Every paired speaker-bootstrap interval for WavLM minus AST contains zero, and so does every one for BiGRU except recall, where WavLM is higher (Section 9). With 13 speakers the data cannot do more. Only LR is clearly worse.
3. **LR is clearly the weakest.** Its EER is {p(raw['LR']['operating']['eer'])} and it detects {p(raw['LR']['operating']['rec_xtts'])} of unseen XTTS at 0.5. It is not competitive.
4. **WavLM has the best probability quality** under the common calibration: lowest log loss, Brier score and ECE (Section 7).
5. **WavLM is the most sensitive and least specific model once thresholds are chosen on the operating role.** At the contract threshold it detects {p(d['thresholds_applied_to_operating']['WavLM']['operating']['rec_xtts'])} of XTTS but flags {p(d['thresholds_applied_to_operating']['WavLM']['operating']['fpr'])} of genuine recordings, against AST's {p(d['thresholds_applied_to_operating']['AST']['operating']['rec_xtts'])} and {p(d['thresholds_applied_to_operating']['AST']['operating']['fpr'])}. AST's operating macro F1 ({f4(d['thresholds_applied_to_operating']['AST']['operating']['macro_f1'])}) is slightly higher than WavLM's ({f4(d['thresholds_applied_to_operating']['WavLM']['operating']['macro_f1'])}) and it does so with far fewer false alarms.
6. **WavLM does not depend on the duration shortcut.** On the duration-matched selection subset it is first or tied on every measure at raw 0.5 (Section 11).
7. **Deployment qualification differs.** On the development CPU, AST needs an estimated 13.7 s for a ten-second recording and WavLM 4.1 s, against the contract's five-second target. LR and BiGRU have not been timed (Section 4).
8. **The false alarms are speaker-specific for every model**, and the problem speakers differ by model. Speaker 6476 is hard for WavLM, AST and LR. Speaker 6454 is a BiGRU-only problem (Section 10).

## 3. Calibration status of each model

| Model | Calibration so far | Status |
|---|---|---|
| WavLM | Affine on log-odds (a = 1.108, b = 0.917), accepted by the pre-declared rule; deployed threshold 0.4040 on the calibrated score | Complete, verified, reproduced on Colab. Sits outside the GR3 lock. |
| LR | Temperature scaling, reported by its owner as the team decision | Complete. It leaves LR badly miscalibrated (Section 7). |
| BiGRU | Platt and smoothed isotonic, investigated by its owner | The owner's report says this is not a final team decision. |
| AST | None | No owner or work found. |

The mappings the common reference fitted on each model's own calibration role:

{params_table()}

A temperature above 1 means a model is overconfident, and below 1 underconfident. LR and BiGRU are overconfident, AST is slightly underconfident, and WavLM is almost exactly right (T = 1.044). The affine offset b is positive for all four (0.73 to 1.04), the same order as ln 3 = 1.10, the log-odds of a 75% TTS prior. This is consistent with the validation prevalence driving the offset, although I did not isolate that. **Either way, the fitted offset describes the validation roles, not real-world prevalence.**

## 4. Ranking under the contract rule and deployment qualification

The contract ranks deployable candidates by unseen-generator selection-role macro F1 at 0.5, then lower EER, then lower genuine FPR at 0.5, then lower warm end-to-end p95 latency.

{rank_table()}

The first criterion separates all four, so the tie-breaks do not come into play.

**Latency.** Only AST and WavLM have been measured, on one development machine (Windows, Intel CPU, fp32, 4 threads, synthetic noise, batch 8, excluding audio decoding). The contract requires a declared benchmark, warm-up and repetitions. These numbers are a preliminary estimate, not the R3-N2 qualification.

| Model | Per 4 s chunk, batch 1 | Estimated 10 s recording | Against the 5 s target |
|---|---:|---:|---|
| WavLM | 1.03 s | 4.1 s | Under (estimate) |
| AST | 1.92 s | 13.7 s | Over on this CPU |
| BiGRU | not measured | not measured | open |
| LR | not measured | not measured | open |

Source: `reports/r3/r3_u2_latency_cpu_fp32_20261112T195158Z.json`. AST would need a GPU or a different deployment target to meet the target.

## 5. Discrimination at raw score 0.5, all three roles

Each table uses raw model scores with the plain 0.5 threshold, so no calibration or threshold choice is involved.

### Selection role (25 speakers, 2,000 recordings)
{role_table('selection')}

### Calibration role (12 speakers, 960 recordings)
{role_table('calibration')}

### Operating role (13 speakers, 1,040 recordings)
{role_table('operating')}

**Reading it.**
- WavLM has the highest ROC-AUC in all three roles ({f4(raw['WavLM']['selection']['auc'])}, {f4(raw['WavLM']['calibration']['auc'])}, {f4(raw['WavLM']['operating']['auc'])}), with AST close behind.
- On the operating role WavLM's EER ({f4(raw['WavLM']['operating']['eer'])}) is worse than AST's ({f4(raw['AST']['operating']['eer'])}) and about the same as BiGRU's ({f4(raw['BiGRU']['operating']['eer'])}). On the selection role, with 25 speakers, it is clearly the best ({f4(raw['WavLM']['selection']['eer'])}). The operating ordering rests on 13 speakers and is noisier.
- WavLM's genuine FPR at 0.5 rises from {p(raw['WavLM']['selection']['fpr'])} (selection) to {p(raw['WavLM']['operating']['fpr'])} (operating). That change comes from the operating speakers, and Section 10 shows which ones.

## 6. Recall by generator (raw 0.5)

{gen_table()}

- **Every model catches essentially all MeloTTS and VITS** (LR is the exception on VITS, about 90 to 91%).
- **Unseen XTTS is the whole difficulty.** At 0.5 WavLM, AST and BiGRU detect between {p(raw['BiGRU']['operating']['rec_xtts'], 0)} and {p(raw['WavLM']['operating']['rec_xtts'], 0)} of it on the operating role. LR detects {p(raw['LR']['operating']['rec_xtts'], 0)}.
- WavLM and AST are within one percentage point of each other on XTTS at 0.5. Whatever separates them lies elsewhere.

## 7. Probability quality on the operating role

### 7.1 One common reference calibration, fitted on each model's own calibration role

Temperature scaling and the affine log-odds mapping, both fitted on the calibration role only and applied unchanged to the operating role. Lower is better for all three measures.

{prob_table()}

- **WavLM is best on every measure, both raw and after calibration.** Its raw scores were already the best calibrated, which matches its temperature of 1.044.
- **The affine mapping helps all four models** by more than temperature scaling does. It helps LR and BiGRU most (log loss drops {f3(d['common_calibration_operating']['LR']['raw']['logloss'] - d['common_calibration_operating']['LR']['affine']['logloss'])} and {f3(d['common_calibration_operating']['BiGRU']['raw']['logloss'] - d['common_calibration_operating']['BiGRU']['affine']['logloss'])}).
- **Temperature scaling does not fix LR.** Its ECE stays near 0.21, so the mapping family LR was given cannot correct its shape.
- **After affine calibration the models rank WavLM, then AST, BiGRU and LR** on all three measures.

### 7.2 The owners' own calibrations (operating role)

{own_table()}

These use different methods and are shown for context only. BiGRU's Platt mapping reaches log loss {f4(d['teammate_calibrations_operating']['BiGRU, Platt']['logloss'])}, better than the common affine mapping on BiGRU ({f4(d['common_calibration_operating']['BiGRU']['affine']['logloss'])}) but still behind AST ({f4(d['common_calibration_operating']['AST']['affine']['logloss'])}) and WavLM ({f4(d['common_calibration_operating']['WavLM']['affine']['logloss'])}) under the common mapping. LR's temperature scaling leaves it far behind.

## 8. Operating thresholds (contract rule: maximise macro F1)

The contract's threshold rule is monotone-invariant, so the decisions below are the same whether the threshold is applied to raw or calibrated scores. All three tables score the same operating-role recordings. They differ in where the threshold was chosen.

### 8.1 Threshold chosen on the operating role (the contract's rule, optimistic)
{thr_table('operating')}

This is the number each model would report under the contract. Because the threshold was tuned on the same recordings it is scored on, it slightly flatters every model.

### 8.2 Threshold chosen on the calibration role (not seen by the operating role)
{thr_table('calibration')}

### 8.3 Threshold chosen on the selection role (25 speakers)
{thr_table('selection')}

**Reading it.**
- **AST has the best balance in all three versions.** Its false-alarm rate stays at 7.3% and its macro F1 stays between 0.914 and 0.923.
- **WavLM trades false alarms for recall.** It gives the highest recall (96 to 98%) and the highest XTTS recall (89 to 94%), at 17.7 to 19.2% false alarms. Its macro F1 is comparable to AST's (0.903 to 0.913 against 0.914 to 0.923).
- **BiGRU sits between them** (12 to 16% false alarms, F1 0.895 to 0.908).
- **LR is far behind** whichever version is used.
- **The macro-F1 rule rewards recall because validation is 75% TTS.** If real TTS prevalence is lower, a high-recall, high-false-alarm profile would cost more than these numbers show, which would favour AST's profile.

## 9. Uncertainty: speaker bootstrap on the operating role

Speakers are resampled with replacement (13 operating speakers, 1,000 resamples, seed 42). All figures use raw scores at 0.5.

{ci_table()}

Paired differences, WavLM minus each model, on the same resamples (an asterisk marks an interval that excludes zero):

{paired_table()}

- **Only LR is separated from WavLM.** For AST and BiGRU every interval for the discrimination measures contains zero.
- **WavLM has higher recall than BiGRU at 0.5** (the interval excludes zero), but not significantly higher than AST's.
- **LR's low false-alarm rate at 0.5 is not a strength.** It reflects a model that rarely flags anything (recall {p(raw['LR']['operating']['recall'], 0)}).
- **The intervals on false-alarm rate are very wide for every model** (up to 0 to 22%). Thirteen speakers cannot support a firm claim about the false-alarm rate of any of them.

## 10. False alarms by speaker (threshold chosen on the operating role)

{conc_table()}

- **WavLM:** four of 13 speakers (3699, 6476, 4406, 5561) hold 47 of 49 false alarms. Eight speakers have none.
- **AST:** one speaker, 6476, holds 15 of its 19.
- **BiGRU:** it flags all 20 genuine recordings of speaker 6454, which is 20 of its 36 false alarms.
- **LR:** false alarms are spread over 10 speakers.
- **Speaker 6476 is hard for WavLM, AST and LR.** Speaker 6454 is a BiGRU-only problem. Speakers 3699, 4406 and 5561 are flagged mostly by WavLM (BiGRU flags a few of 4406 and 3699).
- **A listening check of the WavLM false alarms is under way** (see Section 13). It matters for how this table is read.

## 11. Robustness to the duration shortcut (selection role, duration-matched subset)

The R2-S3 probe found a duration shortcut in the development data. This subset (810 recordings) is matched on duration, so duration cannot help. Raw scores at 0.5:

{dm_table()}

- **WavLM is first or tied on every measure** (recall and XTTS recall tie with BiGRU). Its ROC-AUC is almost unchanged from the full role ({f4(d['duration_matched_selection_raw_0_5']['WavLM']['auc'])} against {f4(raw['WavLM']['selection']['auc'])}).
- AST and BiGRU lose a little ranking quality (AST AUC falls from {f4(raw['AST']['selection']['auc'])} to {f4(d['duration_matched_selection_raw_0_5']['AST']['auc'])}), although macro F1 at 0.5 rises for both on this subset.
- A model leaning on duration would be expected to lose ranking quality when duration is matched. WavLM's AUC barely moves, so at raw 0.5 there is no sign that it leans on duration. The calibrated WavLM policy is a different matter and was shown separately: the deployed rule does worse on this subset than raw 0.5.

## 12. Agreement between models and a descriptive ensemble

### 12.1 Agreement on the operating role (raw 0.5)

{agree_table()}

- **AST and WavLM agree most** (94% of decisions). They are the two models whose decisions are most alike.
- Rank correlations of 0.80 to 0.85 mean the models share a lot of information but are not copies of each other.

### 12.2 Unweighted mean of raw scores (descriptive only)

{ens_table()}

The plan defers ensembles (§9 and §15). These rows are not a candidate, only an indication of what combining scores might offer.

- **Averaging gives small gains in ranking, not a step change.** WavLM and AST together reach ROC-AUC {f4(d['descriptive_unweighted_mean_ensembles_operating']['mean of WavLM and AST']['auc'])}, against {f4(raw['WavLM']['operating']['auc'])} for WavLM alone, and false alarms fall to {p(d['descriptive_unweighted_mean_ensembles_operating']['mean of WavLM and AST']['fpr'])} at 0.5.
- Averaging uncalibrated scores from models with different scales is crude. A proper combination would calibrate each score first.

## 13. Interpretation and limits

**Where WavLM stands.**
- **Strengths:** best ranking on the selection role, best probability quality, robust to the duration shortcut, and the lowest estimated CPU latency of the two measured transformer models.
- **Weakness:** at the contract threshold it flags {p(d['thresholds_applied_to_operating']['WavLM']['operating']['fpr'])} of genuine operating recordings, mostly from four speakers. AST achieves a similar macro F1 with {p(d['thresholds_applied_to_operating']['AST']['operating']['fpr'])}.
- **What would change the picture:** if the audited genuine recordings that WavLM flags are in fact synthetic or processed (Subhan reports hearing TTS-like quality in them; this is user-reported and not yet a recorded audit), part of that false-alarm rate is a labelling problem. The cross-model check supports this only partly. The other models score those recordings higher than other genuine speech (ROC-AUC 0.77 to 0.83) but rarely flag them (0 to 22%), and only speaker 6476 is flagged by several models. This is unresolved.

**What this report cannot show.**
- **It is not an independent confirmation of the GR3 ranking.** The ranking under the contract is shown, but the lock, the deployment qualification and the production decision are separate GR3 steps that have not happened.
- **Thirteen operating speakers are few.** Differences of a few points in macro F1 or EER between WavLM, AST and BiGRU are within noise. Speaker-level effects (Section 10) are large relative to the model differences.
- **All figures are development results.** Thresholds chosen on the operating role are optimistic. No test panel, corruption condition or Protocol A model was touched.
- **The known-generator panel is not compared.** The earlier WavLM result there (the deployed rule scoring macro F1 0.9255 against 0.9645 for raw at 0.5) was computed for WavLM only.
- **Latency is partial** (AST and WavLM only, one machine, synthetic input).
- **AST has no calibration owner,** so its probabilities here come from the common reference mapping, not a team-approved one.
- **Validation prevalence is 75% TTS.** Calibrated probabilities and the macro-F1 thresholds both assume it. They should not be quoted as real-world rates.

**Decisions this report points to, for the team.**
1. Whether the contract's macro-F1 threshold is acceptable given WavLM's false-alarm profile, or whether an amendment (prevalence correction, FPR cap or abstention) is wanted before Protocol A.
2. Who calibrates AST, and with which method, so GR3 compares four calibrated candidates.
3. The outcome of the listening audit of the flagged genuine recordings, recorded as user-reported evidence.
4. Latency measurement for LR and BiGRU, and a GPU or alternative deployment target for AST if it is to qualify.

## 14. Reproduction

Inputs (first 16 hex digits of each file's sha256):

{hash_lines}

The calibration run `runs/r3/r3_n1_wavlm/20261114T083533Z-e6ff5a288068` provides the WavLM raw scores for the cross-check; they equal the R3-U1 export exactly (checked: {d['wavlm_raw_equals_calibration_run_raw']}).

Method notes:
- Metrics use the project's `compute_metrics` and `select_threshold` (contract tie-breaks) and the affine fit and ECE code from the WavLM calibration work. ECE uses 10 equal-width bins.
- The common temperature is fitted by bounded 1-D minimisation of log loss. The affine mapping uses the same settings as the WavLM calibration (regularisation 0.01).
- Bootstrap: 1,000 speaker resamples, seed 42, the same resample for every model so paired differences are paired.
- Run from the project root (numpy, scipy, pyarrow only). The scripts are in `scripts/r3_model_comparison/`: `compare_models.py <out.json>` computes every figure, `make_report.py <out.json> REPORT.md` writes this report from that file, and `wavlm_false_alarm_analysis.py` and `wavlm_false_alarm_cross_model.py` produce the speaker and cross-model false-alarm figures quoted in Sections 10 and 13. The saved computation output is `reports/r3/model_comparison/compare.json` (git-ignored; it travels by Drive).
"""
open(OUT, 'w', encoding='utf-8', newline='\n').write(TEXT)
print('written', OUT, len(TEXT))
