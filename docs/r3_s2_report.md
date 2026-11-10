# R3-S2: report pages 3 to 6 (data, cleaning, EDA, methodology, grouping, protocols, leakage)

Status: **first draft written, not reviewed.** Owner: Subhan, with Muneeb. The plan assigns the review of pages 3 to 6 to Usman. Draft location: `reports/final/`.

## What is there

| File | Role |
|---|---|
| `reports/final/pages_3_6_template.md` | The text. **Edit this file.** Data numbers are `{{placeholders}}`. |
| `reports/final/pages_3_6_draft.md` | The rendered draft. Generated; never edit by hand. |
| `reports/final/pages_3_6_facts.json` | Every number used, with its source file and whether it was read or recomputed. |
| `reports/final/figures/methodology_diagram.{png,svg}` | The methodology diagram (page 5). |
| `reports/final/figures/split_composition.{png,svg}` | Recordings per partition and role for Protocols B and A (page 5). |
| `src/echoproof/reporting/{facts,render,figures}.py`, `scripts/r3_s2_report.py` | The code that builds the above. |
| `tests/test_r3_s2_reporting.py` | 12 tests. |

Page 4 also uses the existing EDA figures in `reports/eda/` and the shortcut-probe figure in `reports/r2/`, referenced by relative path. They must be copied or linked when the report is assembled.

## How it is built, and why

Every number about the data comes from a project file, not from memory. `compute_facts` reads or recomputes about 150 values (cohort counts from the manifest, split counts from the split report, EDA statistics from the measurement table, probe results from the probe JSON, corruption evidence from the version 2 Colab report). Where two sources describe the same quantity it checks that they agree and stops if they do not; for example it raises an error if the leakage report records any violation, if the cohort has a sample rate other than 16 kHz, or if the WavLM cache contains ChatterboxTTS audio, since the text claims otherwise. The renderer fills the template and refuses any placeholder that is not a known fact.

`test_generated_draft_matches_template_and_sources` fails if the draft was edited by hand, or if the template or any source changed without regenerating. Regenerate with:

    python scripts/r3_s2_report.py all

## Choices the draft makes that are not yet recorded elsewhere

These are statements of intent in the draft. They should be confirmed by the team and, where they bind later work, entered in the decision log.

- Every Protocol B model is to be reported on both the full selection set and the duration-matched set (from the R2-S3 probe memo, not yet in the decision log).
- Protocol C (external transfer) is stated as not part of this delivery. The revised plan neither lists it as mandatory nor as cut, so this is an inference; the plan's list of cut items should name it.
- The limitations list (6.4) states that near-duplicate search, multi-seed repeats, nested cross-fitting and Protocol C are not done, following the plan's cut list.

## Things the draft discloses on purpose

- The WavLM embedding cache holds embeddings of the Protocol B test genuine recordings (an acknowledged extraction-only exception in the decision log); nothing was scored.
- Only a provisional lock of the logistic-regression bundle exists; the full Protocol B lock (GR3) has not been made, and the Protocol A exposure history is recorded as unconfirmed.
- Source gaps from Phase 2: deduplication of the full release not run, archive checksums not compared with the publisher's, language not independently verified, relationships between a stem's recordings unverified.
- The 64 kbps to 32 kbps change to the corruption suite was made after looking at validation audio (no model score involved).

## Not done

- No layout into pages, and no check that each section fits one page.
- No review by Usman or Muneeb.
- Pages 3 to 6 do not report any model result. Those belong to pages 7 to 11, which others own.
