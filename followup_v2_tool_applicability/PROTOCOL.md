# Matched tool-applicability pilot

The original study compared two interface packages on requests drawn from different BFCL category files. This follow-up asks whether the direction of the false-call difference persists when applicability changes within a sentence. The bank contains 20 independently authored mock minimal pairs, balanced across five tool families. All 40 requests concern benign, fictitious tools and places. Each evaluated pair changes a single scope word, with the expected decision changing from one specified call to abstention. Two distinct smoke controls are excluded from every estimate.

## Conditions and endpoints

Run every request on both cached Qwen3-4B and Qwen3.5-4B GGUF models. Compare a text JSON catalogue with native tool declarations via the same local llama.cpp chat-completions API and each model's own template. The catalogues have identical tool schemas and names. The task rule, real user message, seed, greedy settings, and output cap are shared. Tool delivery and output protocol differ together. The run is an interface-package comparison.

- Primary endpoint is any recognizable call attempt on an out-of-scope request, regardless of argument or format quality. Compare native minus text in paired percentage points for each model. Report the four-cell paired table and exact two-sided sign test of discordances.
- Secondary endpoints are exact abstentions, semantically correct calls, calls in the required canonical format, and a scope pair passed only when the model makes the exact relevant call and explicitly returns NO_CALL for the matched irrelevant request.
- Semantics evaluate exact tool name and arguments, including numeric equality and case-preserving values. A complete recognizable call with protocol defects is still counted as an attempt. Record raw responses, finish reasons, parse paths, and per-case scores. Output-cap completions stay in the denominator. Server errors are explicit and prevent a completed-pilot claim when any case is missing.
- Report a fixed-family weighted pair bootstrap descriptor with 10,000 resamples, seed 20261120. Repeated tool families and the small authored bank limit inferential scope. This interval describes sensitivity to sampled pairs within this bank.

The runner sends serial requests within each model and runs the two models in parallel. It shuffles case order by fixed seed and reverses interface order for half the cases. The planned inference has 160 main requests and 8 smoke-control requests. There is no tool executor, network target, external tool action, or activation intervention. The study tests a new behavior bank. It does not isolate a circuit, recreate BFCL, assess general agent safety, or establish external validity.
