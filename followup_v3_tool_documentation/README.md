# When a tool’s documentation address matches the request

## Test on external documentation

With published descriptions and original names, Qwen generated a schema-valid call to a distractor in 3/60 presentations and SmolLM2 in 24/60. Across 12 external location questions, changing only the office city produced one fewer distractor attempt in each model. Full instructions, independently authored descriptions and all model replies are available. [External study and figures](independent_descriptions/README.md) · [Blinded responses](independent_descriptions/TARA_Blind_Review.zip)


A model answering a request for East should abstain if a service covers only West, even when its manual was written in East. This experiment changes those two addresses independently and checks both the decision and the arguments of the resulting call.

[Research paper](manuscript.md) · [Supplementary methods and results](supplement.md) · [Original TARA participant study](https://github.com/MateenahJAHAN/tara-chat-tool-study/tree/b0676becc8354be9849c165db6276f35ce0709ca)

## Two checkpoints, the same 512 requests

Changing only the office district converted 70/128 Qwen2.5 decisions from abstention to a false call. SmolLM2 attempted calls in both conditions on all 128 pairs. Both checkpoints returned the correct arguments for all 256 in-scope requests.

![Attempted calls by model, vocabulary, field order and district match](calls_by_scope.png)

[Qwen generations](transfer_01/raw.jsonl) · [SmolLM2 generations](replication_01/raw.jsonl) · [All conditions](paper_cells.csv) · [Paired changes](paper_pairs.csv) · [Individual services](paper_tools.csv) · [Independent response audit](paper_behavior_audit.json)

## Coverage judgments and interventions

The Qwen study also measured how changing one district word, or swapping its activation after block 14, affected the model’s explicit coverage judgment. Its native tool template assigned negligible probability to the requested answer letters. Free responses are scored separately.

![Joint comparison of word changes and single-site activation interventions](field_and_patch.png)

[Prompt-level logits](run_01/raw.jsonl) · [Initial action responses](actions_01/raw.jsonl) · [Service contrasts](family_effects.csv) · [Control checks](independent_checks.json) · [All judgment and action estimates](results.json)

## Does the activation swap change a generated call?

Among 32 out-of-scope requests where only the office matched, swapping the office input embedding prevented 32 false calls. A block-14 swap prevented none. Both block-14 field swaps preserved all 64 correct in-scope calls. One false call changed its district argument. All 512 archive, identity and embedding-control comparisons reproduced the expected token sequence.

![Call attempts after the two field interventions](action_intervention/calls_after_patch.png)

[Protocol](action_intervention/PROTOCOL.md) · [Inference code](action_intervention/run_generation_patch.py) · [Full generations](action_intervention/run_01/raw.jsonl) · [Cell results](action_intervention/cells.csv) · [Verification](action_intervention/verification.json) · [Recount and figure code](action_intervention/analyze_generation_patch.py)

## Code and study design

- [Service definitions](bank_spec.json), [judgment and patch protocol](PROTOCOL.md), [action protocol](ACTION_PROTOCOL.md), [vocabulary and order protocol](TRANSFER_PROTOCOL.md) and [second-checkpoint protocol](REPLICATION_PROTOCOL.md).
- [Judgment and patch runner](run.py), [generation and scoring](run_actions.py), [Qwen vocabulary transfer](run_transfer.py), [SmolLM2 replication](run_replication.py) and [dependency versions](requirements.txt).
- [Judgment analysis](analyze.py), [independent intervention recount](verify_results.py), [Qwen transfer analysis](analyze_transfer.py), [cross-checkpoint response audit](audit_paper.py) and [paper figures](paper_figures.py).
- [Qwen judgment manifest](run_01/run.json), [Qwen action manifest](actions_01/run.json), [Qwen transfer manifest](transfer_01/run.json) and [SmolLM2 manifest](replication_01/run.json).
- [Recount of the original TARA archive](prior_verification.json) and [archive verification code](audit_prior.py).

## Reproduce the experiments

With the pinned weights in local directories, the following commands run inference into fresh output directories. The independent-checkpoint runner reads the hash-verified original 512 rendered prompts. The analysis commands reproduce counts and figures from the stored raw outputs.

```sh
python run.py --model "$QWEN_DIR" --out repeat_judgments
python run_actions.py --model "$QWEN_DIR" --out repeat_actions
python run_transfer.py --model "$QWEN_DIR" --out repeat_transfer
python run_replication.py --model "$SMOL_DIR" --out repeat_smol
python action_intervention/run_generation_patch.py --model "$QWEN_DIR" --out repeat_patch
python analyze.py
python verify_results.py
python audit_paper.py
python paper_figures.py
python action_intervention/analyze_generation_patch.py
```
