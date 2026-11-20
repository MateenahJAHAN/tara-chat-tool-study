# Real-weight development calibration

This bundle contains actual local Qwen2.5-1.5B-Instruct generations and residual activations on two predeclared public BFCL native-training items. It is development/runtime calibration, not the main TARA experiment. No held-out result, effect estimate, human rating or safety conclusion is reported.

What ran:

- Four greedy generations: two official native-training IDs across text-catalogue and native-tool encodings.
- Eight explicit plain/hooked prefill control forwards; 112 finite residual vectors from 28 decoder layers, width 1536.
- Float32/MPS passed exact non-modifying-hook logit identity (maximum difference 0) in this run.
- The earlier float16/MPS attempt produced non-finite values and stopped before generation. A separately frozen six-forward numerical diagnostic retained both precision outcomes before selecting float32. This is a finding about this configuration, not all hardware/models.

Observed examples:

| Item | Text catalogue | Native tool template |
| --- | --- | --- |
| simple_python_0 | Expected function/arguments emitted; closing tool tag missing | Expected function/arguments in a complete tool block |
| irrelevance_0 | NO_CALL | Attempted a function absent from the supplied catalogue |

These are individual development examples, not rates or evidence that one interface is generally better. Raw strings/token IDs are retained so call attempts, formatting and catalogue validity can be separated. No generated function was executed.

## Files

- results/raw_generations.jsonl: four actual raw generations, token IDs, full rendered prompts and hook metadata.
- results/last_input_token_residuals.npz: 112 numeric vectors.
- results/numeric_diagnostic_result.json and raw_numeric_diagnostics.jsonl: precision-control measurements. Tiny negative computed KL values near zero are floating-point error; raw values are retained.
- results/retained_float16_failure.json: failed development attempt.
- protocol/: pre-generation development choices.
- provenance/: original local freeze hashes, pinned source/model manifests and runtime versions. Historical freeze paths describe the original operator layout; they are not a claim this portable copy was itself run.
- scripts/: original portable-path adaptations of the executed operator logic. These adaptations have static/source checks but have not been rerun for this bundle.
- data/: the two original public benchmark rows; source categories remain the benchmark's labels, not invented human annotations.

## Reproduction

Tested runtime: Apple arm64/MPS, Python3.12.11, PyTorch2.15.0, Transformers5.2.0, float32. Other devices/backends are not validated. Model weights are about3.1GB and are intentionally absent.

1. Create a virtual environment with Python3.12 and install requirements.lock.txt.
2. Run scripts/download_model.py with that interpreter. It downloads the exact ungated Apache2 model revision with token=False and no remote custom code.
3. Run scripts/reproduce_numeric_diagnostic.py --output-dir runs/numeric_001.
4. Run scripts/reproduce_calibration.py --output-dir runs/calibration_001.

Choose new output directories. Do not overwrite retained results. The public artifact manifest is verified before reproduction. Downloads and fresh reproductions may consume local disk/compute; no external API/function is executed by the generated model output.

## Limits and next phase

Both IDs, their query groups and the listed function schemas are development-only and must be excluded from final heldout. The source categories differ in provenance; a later semantic probe can learn source/style artifacts. The planned substantive study needs group-disjoint/cross-source evaluation, source/length/schema/lexical controls, dev-only layer/steering selection and a prospective heldout gate. Public benchmark contamination is possible.

No classifier/steering accuracy or main inference is claimed here. Course methods are acknowledged in NOTICE; no unlicensed course implementation is copied. See the main repository's current study status before treating any work as completed research.
