# AST alternative-calibration exploratory results

Run: `20261114T112414Z-7acaafec`. No reliable improvement established under all predeclared gates. Nothing accepted or promoted.

11 fixture tests passed. All folds contain both classes. All 19 completed-run manifest entries verified; original protected artifacts unchanged. The incomplete first run is retained separately.

## Design and limits

Calibration LOSO (12 folds); calibration grouped 4-fold and pooled grouped 5-fold, each at seeds 42/43/44. Equal recording weights. Paired speaker bootstrap: 1,000 resamples, seed 42. NLL/Brier and both ECE differences are in each results.json. Pooled and operating outcomes are exploratory, not independent evidence. Fixed-prediction bootstrap does not incorporate calibration-refitting uncertainty, overlapping CV training sets, or multiplicity. No thresholds or calibrators were selected for official use.

Equal-count ECE uses stable sorting then ten near-equal-sized bins; ties can straddle bins. Equal-width ECE uses ten fixed bins, including 1 in the last. Reliability tables are saved for every candidate and analysis.

## calibration_4fold_42

| Method | NLL | Brier | ECE width | ECE count | AUROC |
|---|---:|---:|---:|---:|---:|
| raw | 0.180424 | 0.054938 | 0.054938 | 0.055784 | 0.982292 |
| temperature | 0.179155 | 0.054064 | 0.053891 | 0.053891 | 0.982112 |
| platt | 0.156458 | 0.049130 | 0.024232 | 0.017231 | 0.980938 |
| beta | 0.150004 | 0.047734 | 0.016741 | 0.012097 | 0.980874 |
| isotonic | 0.172407 | 0.047924 | 0.021720 | 0.007580 | 0.978895 |

NLL difference 95% speaker-bootstrap intervals (candidate minus raw):

- temperature: [-0.003738423345062003, 0.0019192919898264398]
- platt: [-0.044911985296949175, 0.007017088268631886]
- beta: [-0.05173149711986153, 0.0054263687923953915]
- isotonic: [-0.054547515235702504, 0.0631747618012105]

Parameters and per-fold results, exact speaker membership, reliability bins, and per-speaker influences: `runs/20261114T112414Z-7acaafec/calibration_4fold_42/results.json`.

## calibration_4fold_43

| Method | NLL | Brier | ECE width | ECE count | AUROC |
|---|---:|---:|---:|---:|---:|
| raw | 0.180424 | 0.054938 | 0.054938 | 0.055784 | 0.982292 |
| temperature | 0.178759 | 0.053968 | 0.053937 | 0.054336 | 0.982263 |
| platt | 0.153571 | 0.047978 | 0.026319 | 0.017126 | 0.981696 |
| beta | 0.146723 | 0.046693 | 0.009054 | 0.004904 | 0.981470 |
| isotonic | 0.184529 | 0.047443 | 0.017234 | 0.005494 | 0.978822 |

NLL difference 95% speaker-bootstrap intervals (candidate minus raw):

- temperature: [-0.004467230176634914, 0.0016320346500655859]
- platt: [-0.047196239542889235, 0.001720056798034732]
- beta: [-0.05410724025550495, -0.00128147580571888]
- isotonic: [-0.04745819015832515, 0.07605775194669136]

Parameters and per-fold results, exact speaker membership, reliability bins, and per-speaker influences: `runs/20261114T112414Z-7acaafec/calibration_4fold_43/results.json`.

## calibration_4fold_44

| Method | NLL | Brier | ECE width | ECE count | AUROC |
|---|---:|---:|---:|---:|---:|
| raw | 0.180424 | 0.054938 | 0.054938 | 0.055784 | 0.982292 |
| temperature | 0.178765 | 0.053957 | 0.053965 | 0.054374 | 0.982280 |
| platt | 0.155320 | 0.048685 | 0.022394 | 0.019321 | 0.981105 |
| beta | 0.147796 | 0.047139 | 0.008341 | 0.010107 | 0.981175 |
| isotonic | 0.170521 | 0.047478 | 0.018932 | 0.009179 | 0.979164 |

NLL difference 95% speaker-bootstrap intervals (candidate minus raw):

- temperature: [-0.004330859124603611, 0.001451930832592916]
- platt: [-0.04519874101263207, 0.005571377875256325]
- beta: [-0.05364810342297911, 0.0013124836439528547]
- isotonic: [-0.05481278695441722, 0.06009613943681024]

Parameters and per-fold results, exact speaker membership, reliability bins, and per-speaker influences: `runs/20261114T112414Z-7acaafec/calibration_4fold_44/results.json`.

## calibration_loso

| Method | NLL | Brier | ECE width | ECE count | AUROC |
|---|---:|---:|---:|---:|---:|
| raw | 0.180424 | 0.054938 | 0.054938 | 0.055784 | 0.982292 |
| temperature | 0.178868 | 0.054012 | 0.053934 | 0.054338 | 0.982234 |
| platt | 0.156012 | 0.048877 | 0.025371 | 0.015678 | 0.980984 |
| beta | 0.149354 | 0.047477 | 0.008581 | 0.009531 | 0.980961 |
| isotonic | 0.172874 | 0.047795 | 0.030996 | 0.012731 | 0.978044 |

NLL difference 95% speaker-bootstrap intervals (candidate minus raw):

- temperature: [-0.00421598058488615, 0.0015763169508400108]
- platt: [-0.04558732916100694, 0.006730892814628148]
- beta: [-0.052796132963598766, 0.00422707044478006]
- isotonic: [-0.05427731859509778, 0.06465934814040653]

Parameters and per-fold results, exact speaker membership, reliability bins, and per-speaker influences: `runs/20261114T112414Z-7acaafec/calibration_loso/results.json`.

## operating

| Method | NLL | Brier | ECE width | ECE count | AUROC |
|---|---:|---:|---:|---:|---:|
| raw | 0.205947 | 0.063323 | 0.062325 | 0.061784 | 0.975582 |
| temperature | 0.210402 | 0.063333 | 0.060515 | 0.060515 | 0.975582 |
| platt | 0.180870 | 0.052532 | 0.028344 | 0.024098 | 0.975582 |
| beta | 0.191688 | 0.052078 | 0.027786 | 0.021416 | 0.975582 |
| isotonic | 0.310742 | 0.052154 | 0.029277 | 0.014667 | 0.960057 |

NLL difference 95% speaker-bootstrap intervals (candidate minus raw):

- temperature: [-0.0035282303413216114, 0.012902624427178712]
- platt: [-0.0508009829909024, 0.01352037037693579]
- beta: [-0.05735545844010517, 0.05659315107968018]
- isotonic: [-0.06441125661635036, 0.3915543551976149]

Parameters and per-fold results, exact speaker membership, reliability bins, and per-speaker influences: `runs/20261114T112414Z-7acaafec/operating/results.json`.

## pooled_5fold_42

| Method | NLL | Brier | ECE width | ECE count | AUROC |
|---|---:|---:|---:|---:|---:|
| raw | 0.193696 | 0.059298 | 0.058429 | 0.058429 | 0.978716 |
| temperature | 0.194835 | 0.059345 | 0.057885 | 0.057885 | 0.978381 |
| platt | 0.169329 | 0.051744 | 0.023224 | 0.017381 | 0.977604 |
| beta | 0.172803 | 0.051119 | 0.024003 | 0.015931 | 0.976739 |
| isotonic | 0.175008 | 0.049464 | 0.014581 | 0.006274 | 0.974848 |

NLL difference 95% speaker-bootstrap intervals (candidate minus raw):

- temperature: [-0.0005390704475048522, 0.003827535734716241]
- platt: [-0.04355740929707515, -0.0011698127599176968]
- beta: [-0.04706440150111899, 0.017078836962001603]
- isotonic: [-0.0580070171801037, 0.03953545798713648]

Parameters and per-fold results, exact speaker membership, reliability bins, and per-speaker influences: `runs/20261114T112414Z-7acaafec/pooled_5fold_42/results.json`.

## pooled_5fold_43

| Method | NLL | Brier | ECE width | ECE count | AUROC |
|---|---:|---:|---:|---:|---:|
| raw | 0.193696 | 0.059298 | 0.058429 | 0.058429 | 0.978716 |
| temperature | 0.194709 | 0.059271 | 0.058052 | 0.058052 | 0.978433 |
| platt | 0.174463 | 0.052677 | 0.023767 | 0.012811 | 0.975793 |
| beta | 0.182632 | 0.052255 | 0.022640 | 0.011943 | 0.974643 |
| isotonic | 0.185649 | 0.051516 | 0.022514 | 0.012847 | 0.971426 |

NLL difference 95% speaker-bootstrap intervals (candidate minus raw):

- temperature: [-0.0008376471284250316, 0.0037225224588564023]
- platt: [-0.04323368192524091, 0.010599189277303299]
- beta: [-0.046738765644760406, 0.04333518240434487]
- isotonic: [-0.054687766784514834, 0.060210331419813024]

Parameters and per-fold results, exact speaker membership, reliability bins, and per-speaker influences: `runs/20261114T112414Z-7acaafec/pooled_5fold_43/results.json`.

## pooled_5fold_44

| Method | NLL | Brier | ECE width | ECE count | AUROC |
|---|---:|---:|---:|---:|---:|
| raw | 0.193696 | 0.059298 | 0.058429 | 0.058429 | 0.978716 |
| temperature | 0.194129 | 0.059220 | 0.058052 | 0.058052 | 0.978571 |
| platt | 0.169216 | 0.051615 | 0.020731 | 0.016644 | 0.977371 |
| beta | 0.173221 | 0.050918 | 0.019711 | 0.011903 | 0.976689 |
| isotonic | 0.174141 | 0.049219 | 0.011391 | 0.006527 | 0.975159 |

NLL difference 95% speaker-bootstrap intervals (candidate minus raw):

- temperature: [-0.0010052655537207437, 0.0025984072194961566]
- platt: [-0.044860802772211196, 8.583000681847606e-05]
- beta: [-0.047768094133100396, 0.020427873220793327]
- isotonic: [-0.0590539051900765, 0.04000130650740691]

Parameters and per-fold results, exact speaker membership, reliability bins, and per-speaker influences: `runs/20261114T112414Z-7acaafec/pooled_5fold_44/results.json`.

## Classification on operating role

Fixed threshold is the unchanged saved value 0.4010431058704853, applied numerically to each transformed score. It is not an equivalent decision boundary after transformation.

| Method | Fixed macro F1 | Accuracy | FPR | FNR | Optimized macro F1 | Optimized threshold |
|---|---:|---:|---:|---:|---:|---:|
| raw | 0.922832 | 0.940385 | 0.073077 | 0.055128 | 0.922832 | 0.401043106 |
| temperature | 0.914823 | 0.933654 | 0.073077 | 0.064103 | 0.922832 | 0.382586687 |
| platt | 0.899046 | 0.925962 | 0.180769 | 0.038462 | 0.922832 | 0.631659629 |
| beta | 0.912821 | 0.934615 | 0.130769 | 0.043590 | 0.922832 | 0.561135275 |
| isotonic | 0.910025 | 0.932692 | 0.138462 | 0.043590 | 0.917099 | 0.552642405 |

Temperature, Platt and beta preserve full-fit operating AUROC and maximum threshold-optimized F1. Their different optimal thresholds are score reparameterizations, not better detection. Isotonic ties reduce AUROC and maximum F1. Pooled OOF AUROC can change even for monotone methods because different folds use different maps.

## Parameter stability

Across all 39 CV fits per method (descriptive, mixing calibration and pooled analyses; separate summaries are in results.json):

| Method | Parameter | Min | Max | Std |
|---|---|---:|---:|---:|
| temperature | temperature | 0.793569 | 1.006597 | 0.065171 |
| platt | a | 1.110983 | 1.522318 | 0.129528 |
| platt | b | 0.898331 | 1.559401 | 0.153185 |
| beta | a | 0.334216 | 0.882564 | 0.166657 |
| beta | b | 1.694753 | 5.178024 | 1.016383 |
| beta | c | -1.779625 | 0.458546 | 0.701983 |
| isotonic | knots | 16.000000 | 40.000000 | 5.059540 |
| isotonic | levels | 8.000000 | 20.000000 | 2.529770 |

No optimizer boundary fits or recorded numerical warnings occurred. All optimizers returned success. A numerical stability acceptance tolerance was not supplied, so stability remains a review gate. Beta varies more than Platt, especially its second slope/intercept. Isotonic full-calibration fit has 20 knots and 10 levels; CV fits have 16?40 knots and 8?20 levels.

## Influence and disagreement

Operating speaker:6476 (80 recordings) has mean NLL changes +0.18285 Platt, +0.37908 beta and +1.74920 isotonic. Omitting that speaker changes overall operating NLL differences to -0.04240, -0.04704 and -0.03224, respectively. This is a sensitivity diagnostic, not permission to exclude the speaker or evidence of corrupt data. Full influence lists and leave-one-speaker-out mean differences are saved.

Beta is strongest on calibration LOSO NLL, but Platt has lower operating NLL. Isotonic improves pooled OOF point estimates but worsens operating NLL markedly; its calibration 4-fold seed 43 NLL also worsens versus raw. Calibration-fold results alone therefore overstate certainty of transfer.

## Decision

- temperature: {"pooled_nll_brier_lower_all_seeds": false, "pooled_nll_ci_below_zero_all_seeds": false, "operating_both_lower": false, "operating_candidate_rule": false, "parameter_stability": "review_required_no_predeclared_tolerance", "verdict": "no reliable improvement established under all required gates; not accepted"}
- platt: {"pooled_nll_brier_lower_all_seeds": true, "pooled_nll_ci_below_zero_all_seeds": false, "operating_both_lower": true, "operating_candidate_rule": true, "parameter_stability": "review_required_no_predeclared_tolerance", "verdict": "no reliable improvement established under all required gates; not accepted"}
- beta: {"pooled_nll_brier_lower_all_seeds": true, "pooled_nll_ci_below_zero_all_seeds": false, "operating_both_lower": true, "operating_candidate_rule": true, "parameter_stability": "review_required_no_predeclared_tolerance", "verdict": "no reliable improvement established under all required gates; not accepted"}
- isotonic: {"pooled_nll_brier_lower_all_seeds": true, "pooled_nll_ci_below_zero_all_seeds": false, "operating_both_lower": false, "operating_candidate_rule": false, "parameter_stability": "review_required_no_predeclared_tolerance", "verdict": "no reliable improvement established under all required gates; not accepted"}

Platt and beta satisfy the operating point-estimate candidate rule, but that is a post-hoc exploratory comparison, not approval. Platt pooled NLL CI excludes zero only at seed 42; beta does not at any seed. Therefore no candidate meets all required gates, regardless of the unresolved stability tolerance.

Recommendation: no automatic calibrator replacement. A separately approved confirmatory follow-up focused on the simpler Platt model is scientifically justified by consistent point-estimate improvements, but requires a predeclared stability criterion and fitting-aware uncertainty or new untouched development groups. Reusing these outcomes is not independent confirmation. Preserve the official temperature-rejection/raw fallback.
