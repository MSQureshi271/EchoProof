# Pinned pretrained models (R1-U1)

Metadata read from the Hugging Face API on 4 November 2026. The revision is the only thing the code accepts: `pinned_snapshot()` in
`src/echoproof/features/pretrained.py` downloads that commit and refuses a snapshot whose directory name differs. Do not use `main`.

| | WavLM-Base-Plus | Audio Spectrogram Transformer (AudioSet) |
|---|---|---|
| Repository | `microsoft/wavlm-base-plus` | `MIT/ast-finetuned-audioset-10-10-0.4593` |
| Pinned commit | `4c66d4806a428f2e922ccfa1a962776e232d487b` | `f826b80d28226b62986cc218e5cec390b1096902` |
| Last modified upstream | 2021-12-22 | 2023-09-06 |
| Weights file | `pytorch_model.bin` | `model.safetensors` |
| Weights SHA-256 | `3bb273a6ace99408b50cfc81afdbb7ef2de02da2eab0234e18db608ce692fe51` (377,617,425 bytes; first verified download, `reports/r1/r1_u1_colab_20261104T114311Z.json`). Later downloads are checked against it. `ae0c1e2ad4e1381d851fa9bf298ba13ebc9c5a914cdee2dbe427a6583869924d` (346,404,948 bytes; first verified download, `reports/r1/ast_probe.json`). Later downloads are checked against it. |
| Licence | **not declared** in the Hugging Face metadata | `bsd-3-clause` (model card metadata) |
| Role | frozen encoder + trained head | head warm-up, then partial fine-tuning of the last two blocks |

## Licence status

- **AST: BSD-3-Clause** as declared in the model card metadata.
- **WavLM-Base-Plus: unresolved.** The Hugging Face repository declares no licence. Before the weights or a derived model are
  redistributed (the plan only requires retrieval instructions and hashes, not redistribution), someone must confirm the terms of
  the upstream WavLM source (`microsoft/unilm`) and record the answer here. Until then this item is open, not assumed.
- Training-source overlap is a separate disclosure: the WavLM model card lists Libri-Light, GigaSpeech and VoxPopuli pretraining. The
  PhonemeDF genuine recordings follow the LibriSpeech naming pattern, so related speech may have been seen in pretraining.

## Preprocessing (WavLM)

The extractor applies exactly what the pinned `preprocessor_config.json` specifies and adds nothing. The expectation recorded in
`PRETRAINED["wavlm"]["expected_preprocessing"]` is 16 kHz, `do_normalize=false`, one channel; the Colab notebook checks the downloaded
file against it and fails loudly if the checkpoint says otherwise.

## Why batches are never zero-padded

The WavLM-Base-Plus front end uses group normalisation over time, so padding a clip with zeros changes its statistics and its
embedding. A sample-level mask cannot be mapped onto hidden frames after that. `WavLMExtractor.embed` therefore batches only clips of
identical length and never passes an attention mask. Short clips are embedded at their true length. Unit tests show the difference
that padding would introduce, and that single and batched results agree within 1e-4.

## Open items

- Both weights hashes are recorded. A snapshot or weights file that differs makes `pinned_snapshot()` fail.
- AST 5.x parameter names (`audio_spectrogram_transformer.layers.N.attention.q_proj…`) differ from 4.x (`…encoder.layer.N.attention.attention.query…`);
  keep transformers pinned (5.22.0 was used for every Colab check) because saved AST checkpoints carry the installed version's names.
- Resolve the WavLM licence question above.
- The AST feature extractor, 128-mel grid and its checkpoint normalisation are checked in R1-U2.
