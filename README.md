# Function applicability across text and tool interfaces

A TARA research project in progress. This repository will contain the prospective protocol, independently implemented code, raw live-model outputs and analysis for a study of function selection and abstention.

## Research question

When a request does not match the available function, does the same model abstain consistently in text-serialized and native tool interfaces? Do linear probes trained in one interface transfer to the other, and do controlled interventions change behaviour while preserving output quality?

## Current status

The [prospective protocol, code, split manifest and tests](study_v1/) are now frozen before main inference, following actual Claude Code Opus 5.5 Max implementation and independent source review. **The substantive held-out experiment has not run yet.** Development is complete; the selected fitted artifacts and the explicit development-discovered measurement amendment are now frozen before held-out execution. Failed checks and negative outcomes will be retained.

The proposed data are the published [Berkeley Function Calling Leaderboard](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard) relevance/irrelevance tasks, with a separate live-data evaluation and controls for source, length and function-schema effects. The candidate open-weight model is Qwen2.5-1.5B-Instruct; exact revisions, licenses and selection rules will be recorded in the protocol.

## Completed development calibration

[Real local model outputs and reproduction code](development_calibration/README.md) are now available: four generations on two declared public BFCL development items, residual activations from all 28 decoder layers and retained precision diagnostics. Float32/MPS passed the non-modifying-hook check; the earlier float16 numerical failure is preserved. These are development checks, not held-out study results.

The [fresh reproduction record](development_calibration/reproduction_R2/README.txt) verifies that the published calibration script reproduces all four original token sequences and all 112 residual vectors exactly.

## Prospective main study

The fixed bank contains 128 training, 32 development and 64 held-out tasks, balanced by source label and drawn from 224 distinct query/function/schema groups. Training uses activation prefills; development and held-out behaviour are separate. The bounded schedule allows up to 512 planned generations plus two numeric-calibration repeats, with a hard cap of 600 attempts. Optional steering remains conditional on the predeclared development gates. The analysis reports finite-bank descriptive estimates and uncertainty.

## Development amendment and held-out freeze

The [V2 measurement amendment](study_v1/MEASUREMENT_AMENDMENT_V2.json) separates semantic call utility from strict tool-tag formatting. Development inspection exposed an implementation error: a complete valid JSON call lacking only its closing tool tag had been treated as semantically incorrect. All 64 original outputs/scores are preserved; separately derived rescoring records are hash-linked. Primary call-attempt counts and strict-format diagnostics did not change. No development inference was repeated.

All probe, surface-control and steering weights were fitted/selected using training and development data only. Text steering failed its development gate. Native steering selected layer 16 and alpha 0.01. These are selection results; the held-out baseline/probe/control and 96 native-steering generations have not run at this commit. The original public tag remains available.

## Methods and scope

The approach draws on linear probes, model steering and agent evaluation in the [ARENA curriculum](https://github.com/ARENA-education/ARENA_materials). The implementation will be original code; course notebooks are not redistributed here.

This is a comparison of interface packages and function applicability. It does not isolate tool availability alone or establish general agent safety, biological-misuse resistance or a clinical result. An earlier six-case fictional-card engineering pilot was a separate connection/control check and will not enter this study's estimates.

## Reproducibility and contributions

Model weights and downloaded benchmark data will stay out of Git. Source revisions, retrieval scripts, dataset/split hashes, commands, resource settings, raw model records and analysis will identify exactly what ran. Upstream models and data retain their own licenses.

The preparation is AI-assisted. Code, study-design and execution assistance from Codex and Claude Code will be documented separately from Mateenah's personal research contribution. This repository is a working study, not a peer-reviewed publication.
