# Phase 1 fixtures

`generated/` is reserved for synthetic software fixtures created by P1-07. Do not put research recordings or transcripts here. Fixture outputs and labels test code only; they do not establish detector accuracy. Fixture registration and content validation are implemented in the later fixture task.

`contracts/` holds fabricated metadata for P1-05. Its placeholder digest is not a checksum of a real recording.

P1-07 creates versioned tone, noise, stereo, silence and invalid-file fixtures under `generated/`, each with a SHA-256 manifest. They are software signals and are not research speech.
