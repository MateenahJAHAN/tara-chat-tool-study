# Function applicability across text and native tool interfaces

A completed exploratory TARA study of Qwen2.5-1.5B-Instruct on public BFCL tasks. It compares function selection and abstention across two interface packages, then tests cross-interface probes and a development-selected steering intervention.

**The real model run, raw outputs, analysis, figures and independent integrity audit are available.** This is an AI-assisted working study, not a peer-reviewed publication.

## Follow-up: matched tool applicability

On a new bank of 20 benign scope pairs, the native interface produced more out-of-scope calls on both tested checkpoints: **14/20 to 19/20** for the original Qwen2.5 1.5B model and **0/20 to 10/20** for Qwen3.5 4B. Canonical formatting, exact abstention, and paired scope accuracy were measured separately. This exploratory result is for the tested prompts and interface packages.

[Follow-up results and figures](followup_v2_tool_applicability/RESULTS.md) · [Frozen protocol](followup_v2_tool_applicability/PROTOCOL.md) · [Machine-readable verification](followup_v2_tool_applicability/verification.json)

![False calls, correct calls, formatting and scope-pair outcomes on the matched bank](followup_v2_tool_applicability/figures/endpoint_counts.png)

[Matched scope-pair design](followup_v2_tool_applicability/figures/study_design.svg)

## Held-out results

The same 64 held-out requests were evaluated with a text function catalogue and the model's native tool template. There were 32 benchmark-labelled irrelevant requests and 32 relevant requests.

| Endpoint | Text catalogue | Native template |
|---|---:|---:|
| False call attempts on irrelevant requests | 9/32 (28.1%) | 16/32 (50.0%) |
| Exact `NO_CALL` on irrelevant requests | 23/32 | 15/32 |
| Semantic reference correctness on relevant requests | 23/32 | 26/32 |
| Canonical call format on relevant requests | 0/32 | 31/32 |

The paired false-call difference was **+21.875 percentage points** for the native interface package. A group-resampling descriptor spans +6.25 to +37.5 points for this selected bank. Two baseline completions reached the token ceiling and remain in the denominators. Semantic correctness and protocol formatting are separate endpoints.

![Held-out behaviour and denominators](reporting/figures_heldout_final/01_behaviour_rates.png)

Frozen logistic probes reached held-out AUROCs of 0.898–0.946 across the two interfaces. Query-only logistic AUROC was 0.747 and the length-only surface baseline reached 0.814. The category source files supply the labels, so source/style cues remain a possible explanation; decodability does not establish a decision mechanism.

The native steering configuration selected on development **did not transfer successfully**: false calls stayed at 2/8 and relevant semantic correctness fell from 7/8 to 6/8. All six scheduled native direction/control configurations are retained. Text steering failed the development gate and was skipped on held-out data.

## Read the evidence

- [Results and limitations](study_v1/RESULTS.txt)
- [Behaviour report](study_v1/reports/behaviour_heldout.json) and [probe/control report](study_v1/reports/heldout_probe_results.json)
- [Raw held-out outputs](study_v1/runs/heldout/) and [development records](study_v1/runs/dev/)
- [Held-out figures](reporting/figures_heldout_final/) and [development figures](reporting/figures_dev_final/)
- [Independent audit: 63 mechanical checks](study_v1/reviews/HELDOUT_INDEPENDENT_AUDIT_PUBLIC.txt)
- [Exact public-file hashes](COMPLETED_PUBLIC_MANIFEST.json)

The audit verified all 224 held-out decode/parser round-trips, planned item banks, finite activation arrays, frozen artifacts, raw counts and ledger totals. It did not independently recompute probe AUROCs or validate interval calibration.

## Design and execution

The pinned bank contains 128 training, 32 development and 64 held-out items from 224 distinct query/function/schema connected components. Training used activation prefills. The complete run retained **416 main generations plus two numeric-repeat calibrations**; 224 generations were held-out baselines/interventions. No external function or API was executed.

- Model: [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct), revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, float32/MPS, eager attention.
- Data: [Berkeley Function Calling Leaderboard](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard), pinned gorilla commit `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8`.
- Methods: paired interface packages, residual-stream mean/logistic probes, query/length/schema/lexical controls, permutation/random controls, and development-gated activation steering.
- Course context: linear probes, model steering and evaluation in the [ARENA curriculum](https://github.com/ARENA-education/ARENA_materials). The implementation is original; course notebooks are not redistributed here.

This estimates an interface-package difference on a selected benchmark bank. Source and category cues, unknown benchmark contamination, and small intervention subsets limit the conclusions. The matched follow-up uses a separate bank and adds a second checkpoint.

## Freeze history and measurement correction

The [V1 protocol/code](https://github.com/MateenahJAHAN/tara-chat-tool-study/tree/v0.1-prospective-plan) were publicly tagged before development. Development inspection found a scorer error: complete valid JSON calls missing only their closing tool tag had been marked semantically incorrect.

The documented [V2 measurement amendment](study_v1/MEASUREMENT_AMENDMENT_V2.json) rescored all 64 development outputs without repeating inference. Original code, outputs, scores and reports are preserved. Primary call-attempt counts and strict-format diagnostics did not change. The amended scorer and fitted artifacts were publicly frozen at [commit 645e581d / tag v0.2-before-heldout](https://github.com/MateenahJAHAN/tara-chat-tool-study/tree/v0.2-before-heldout) before any held-out output. This development-discovered correction is part of the study history.

## Reproduction and provenance

The [operator guide](study_v1/proposal/README.txt), [pinned requirements](study_v1/proposal/requirements.txt), [source/config](study_v1/proposal/), split manifests and model/data hashes identify what ran. Pure scoring and integrity tests can be run with:

```sh
python -m unittest discover -s study_v1/proposal/tests
```

Live reproduction requires the pinned model/data staged locally and a fresh workspace; the recorded held-out gate binds the original frozen artifacts. Model weights and large NPY activation banks are excluded from Git; activation hashes and fitted probes/control weights are retained. The earlier [development calibration](development_calibration/README.md) includes four real generations, an exact reproduction and the retained float16 failure.

## Contributions

Claude Code Opus 5.5 drafted the original design and implementation. Operators reviewed and amended the code, ran the model, scored the outputs, and audited the records. Claude Fable 5.2 reviewed the follow-up design. Mateenah’s individual role in the design, implementation, analysis, and writing remains to be confirmed for a manuscript contribution statement. [Original study provenance](study_v1/AI_ASSISTANCE_PROVENANCE.json) · [Follow-up verification](followup_v2_tool_applicability/verification.json)

Upstream models and benchmark data retain their own licences. This repository makes no official BFCL-score or new-mechanism claim.
