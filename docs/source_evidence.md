# P2-02 source evidence and remaining provenance

## User attestation addendum — 30 October 2026

The user confirms that the Drive files came from Zenodo PhonemeDF release 1.0.1 (record 19522979), no files were changed during installation, and the project has permission to use the downloaded dataset. This confirmation is recorded in [the P2-02 run addendum](../reports/phase2/p2_02-20261030T130521Z-8b4cde3d/user_attestation.md) and reflected in the source registry as user-attested. It does not represent an independent archive checksum match or legal review. The original capture artifacts remain unchanged. P2-03 has not started.

The P2-01 Drive report `20261030T123926Z-6181b135` is accepted for proceeding to source evidence. Its entry hash, collector hash and all five referenced evidence hashes match the supplied files. All four configured dataset roots were present and all seven saved lock artifacts passed. Independent M review remains unrecorded. The report uses Python 3.13.18, whereas P1-13 reproduced 3.13.16: saved lock integrity does not qualify the current scientific runtime. The metadata collector uses only the standard library; later numerical work requires explicit runtime verification/requalification.

## Delivered evidence

- `configs/data_sources.json`: provisional primary/external registry, candidate release and local entry snapshot, component paths/roles and verification flags.
- `reports/phase2/p2_02-preparation-20261030/source_inventory.json`: preparation scope and unresolved items.
- The same directory's `publisher_observations.json`: dated, authored summaries of browsed publisher pages with selected publisher MD5 checksums. These are not raw document downloads; their SHA-256 hashes identify these summaries only.
- `license_inventory.csv`: component-specific pending permissions, observed card terms and required evidence.
- `generator_registry.json`: publisher-supported TTS/VC categories, aliases, unknown versions and local-mapping verification flags.
- `artifact_manifest.json`: preparation artifact hashes. The source inventory separately hashes the registry.

The [PhonemeDF release](https://zenodo.org/records/19522979) identifies v1.0.1 and separate real/XTTSv2/ChatterboxTTS components. Its publisher checksums are MD5. The supplied extracted files have not been matched to those archives or this revision. Dataset revision remains unverified; `phonemedf:drive-entry-20261030T123926Z-6181b135` identifies the administrative local snapshot, not a verified publisher release. Original archive URLs/checksums, retained archives and rename/extraction history remain unresolved.

The [authors' repository](https://github.com/Vamshi-Nallaguntla/PhonemeDF) documents both Zenodo and Kaggle packaging. Its sample naming convention does not establish how the user's extracted paths were produced. Repository commit capture is pending; no commit is fabricated.

The [pinned MLAAD-tiny card](https://huggingface.co/datasets/mueller91/MLAAD-tiny/blob/9143e5ea709575ebab6bec52840a1043aada7bb1/README.md) separates inherited genuine terms from synthetic CC BY-NC 4.0 and describes mixed synthetic languages. The commit identifies documentation to inspect, not acquired dataset audio. English genuine/TTS eligibility, actual counts and inherited `original/LICENSE` remain pending. No source is approved for redistribution through this preliminary ledger.

## Complete raw-document capture in Colab

The local shell's outbound sockets are blocked, and the browser tool could not retrieve API JSON or PDF bytes. Publisher pages were reviewed through browsing, but raw documents and complete license terms could not be saved here. This limitation leaves P2-02 evidence closure pending; it does not block preparing the next configuration task.

1. Sync `configs/data_sources.json`, `scripts/p2_02_source_evidence.py`, and `notebooks/P2_02_Source_Evidence_Colab.ipynb` to Drive.
2. Run the notebook on CPU. No package installation, GPU or P1-03 rerun is required.
3. The collector saves bounded publisher JSON/PDF/README/license files with requested/final URLs, retrieval UTC, byte sizes and SHA-256. It resolves a GitHub commit before requesting the corresponding README, uses the fixed Zenodo record and pinned external documentation revision, and downloads no audio archives.
4. Share the new run's `source_inventory.json`, `artifact_manifest.json` and captured `documents/`. Review actual terms and update registries in a new versioned evidence run. HTTP failures or missing files remain explicit; document retrieval alone does not authorize use or admit recordings.

Acquisition origin, original archives and extraction/renaming history require user clarification if no metadata can recover them. Publisher method categories may guide an adapter but must not turn folder names into verified local labels. ChatterboxTTS keeps its proposed final-family boundary; VC is excluded; MLAAD-tiny audio remains not downloaded.

Next: P2-03 strict audit schemas/configuration can be implemented using provisional source state. Do not call P2-02 fully verified, admit data or pass G2 until the outstanding evidence is resolved and the actual reviewer records a disposition.
