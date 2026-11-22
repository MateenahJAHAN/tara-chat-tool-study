# Does the tool format change when a model calls a tool?

On 32 [Berkeley Function Calling Leaderboard (BFCL)](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard) test requests where no tool applied, Qwen2.5-1.5B-Instruct tried to call a tool **16 times with its built-in tool format and 9 times with a tool list written in the prompt**. A second experiment, using 20 pairs of requests that differ by one word, found more false calls with the built-in format on both models tested.

<a id="held-out-results"></a>

## First experiment · BFCL requests

Both formats used the same 64 test requests. BFCL labels say a listed tool applies to 32 requests and none applies to the other 32.

| Measure | Text list | Built-in format |
|---|---:|---:|
| Tried to call a tool when none applied | 9/32 | 16/32 |
| Returned exactly `NO_CALL` when no tool applied | 23/32 | 15/32 |
| Returned the right function and arguments when a tool applied | 23/32 | 26/32 |
| Used the expected call format when a tool applied | 0/32 | 31/32 |

The built-in format produced about 22 percentage points more false calls. Resampling the test groups gave a range of about 6 to 38 points for this set of requests. Two outputs reached the token limit and are included in the counts. [First experiment results](study_v1/RESULTS.txt)

![False calls and correct calls by tool format](reporting/figures_heldout_final/01_behaviour_rates.png)

<a id="follow-up-matched-tool-applicability"></a>

## Second experiment · Requests that differ by one word

Each pair changes a single word that determines whether a listed tool applies. The 20 pairs use five kinds of fictitious tool.

| Model | False calls with the built-in format | False calls with the text list |
|---|---:|---:|
| Qwen2.5 1.5B | 19/20 | 14/20 |
| Qwen3.5 4B | 10/20 | 0/20 |

[Second experiment results](followup_v2_tool_applicability/RESULTS.md) · [How the requests were paired](followup_v2_tool_applicability/figures/study_design.svg)

![False calls, correct calls and formatting for both models and tool formats](followup_v2_tool_applicability/figures/endpoint_counts.png)

## What the probes and steering showed

Logistic probes on model activations reached test-set AUROCs of 0.898–0.946 across training and test formats. A baseline using only prompt length reached 0.814. BFCL's relevant and irrelevant requests came from different files, so the probes may have picked up differences in wording or style.

The built-in format's steering settings were chosen on development data. On the test subset, false calls stayed at 2/8 and correct calls on relevant requests fell from 7/8 to 6/8. Steering with the text list fell short of the development criteria, so it was tested only on the development set. [Probe and control results](study_v1/reports/heldout_probe_results.json)

The formats differ in how tools are presented and how calls are written. The second experiment used hand-written requests, and the two models ran on different software and at different precision. The results compare formats within each model on these requests. Neither experiment executed the tools.

## Third experiment · Does the documentation office change the call?

In fictional directories, changing only the district where documentation was written induced 70 false calls in 128 paired Qwen requests. That effect did not transfer to the 12 city questions from BFCL and When2Call. Matching the office to the requested city changed Qwen's false-call count from four to three and SmolLM2's from seven to six. With a single unrelated function under its original name, the models supplied schema-valid calls on 3/60 and 24/60 presentations, respectively. No tool was executed. [Study and raw results](followup_v3_tool_documentation/README.md) · [External questions and scoring rules](followup_v3_tool_documentation/independent_descriptions/README.md)

![Wrong-tool calls for the same twelve questions when the documentation office changed](followup_v3_tool_documentation/independent_descriptions/catalogue_office_pairs.png)

## Reproducing the results

- **BFCL experiment** · [Setup guide](study_v1/proposal/README.txt), [requirements](study_v1/proposal/requirements.txt), [scores](study_v1/reports/behaviour_heldout.json), [test outputs](study_v1/runs/heldout/) and [development records](study_v1/runs/dev/). It used [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/tree/989aa7980e4cf806f80c7fef2b1adb7bc71aa306) in float32 on MPS with eager attention, with BFCL at [revision `6ea5797`](https://github.com/ShishirPatil/gorilla/tree/6ea57973c7a6097fd7c5915698c54c17c5b1b6c8). The split used 128 training, 32 development and 64 test requests, grouped by shared query, function or schema.
- **Scoring fix** · A development check found that a complete JSON call missing its closing tool tag was scored as semantically wrong. The fix rescored 64 development outputs. Call-attempt and strict-format counts stayed the same. The fix was [tagged before the test run](https://github.com/MateenahJAHAN/tara-chat-tool-study/tree/v0.2-before-heldout). [Scoring change and affected measures](study_v1/MEASUREMENT_AMENDMENT_V2.json)
- **Paired-request experiment** · The [original protocol](followup_v2_tool_applicability/PROTOCOL.md) specified Qwen3-4B and Qwen3.5-4B. Qwen3-4B reached the length limit on the setup checks, so the [documented amendment](followup_v2_tool_applicability/AMENDMENT_QWEN25.md) replaced it with Qwen2.5-1.5B. [Requests and tool definitions](followup_v2_tool_applicability/bank.json) · [Check results and checksums](followup_v2_tool_applicability/verification.json).
- **Files and checks** · [Independent checks of the saved outputs](study_v1/reviews/HELDOUT_INDEPENDENT_AUDIT_PUBLIC.txt), [file checksums](COMPLETED_PUBLIC_MANIFEST.json), [development figures](reporting/figures_dev_final/) and [early test run](development_calibration/README.md). The checks covered the saved output counts and parsing. Probe AUROC was outside their scope. Full reproduction requires locally staged model weights and data. Fitted probes are included. Checksums identify the large activation files produced during the run.

The scoring tests run without the model.

```sh
python -m unittest discover -s study_v1/proposal/tests
```

The study used methods covered by the [ARENA curriculum](https://github.com/ARENA-education/ARENA_materials). The scores use the project scorer. BFCL publishes its own leaderboard scores. Upstream model and dataset licences apply.

## Use of AI tools

Claude Code Opus 5.5 drafted the original study design and code. The project record documents code review, model runs, scoring and checks. Claude Fable 5.2 reviewed the second experiment's design. [Original study record](study_v1/AI_ASSISTANCE_PROVENANCE.json) · [Second experiment record](followup_v2_tool_applicability/verification.json)
