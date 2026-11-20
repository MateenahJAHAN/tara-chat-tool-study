BFCL interface-package and applicability-probe study — prospective operator-amended version

This is a real public-data/open-weight experiment plan, using pinned Qwen2.5-1.5B-Instruct float32/MPS and BFCL v4 data. The actual Opus5.5 Max lead draft is preserved separately; independent input/scoring/runtime reviews led to pre-outcome operator amendments. The V1 source/input plan was publicly tagged before development. The current V2 scoring amendment was made after64 development outputs and before any heldout result; do not call the entire study immutable or formally preregistered.

Question: on this finite input-selected bank, compare false call attempts across a text catalogue and the model native tool template, then inspect cross-interface/source linear applicability probes with surface/random/permutation controls. This is an interface-package comparison and replication-style behavior baseline. The exploratory extension is controlled probe transfer and gated local steering, not general agent/biology safety. BFCL source categories supply labels and can confound probes; public benchmark contamination is unknown. No function/API is executed.

Input preparation passed: TRAIN64/class (128 rows), DEV16/class (32), HELDOUT32/class (64). Every selected row has a distinct connected-component group. C0 components are excluded from every main bank. All data/model revisions, hashes, IDs, prompts and fixed steering subsets are in config/manifests.

Expected maximum main generation schedule:192 baseline,128 dev steering,192 heldout steering =512. TRAIN uses prefill only. Two separate manual-runtime calibration repeats are permitted; hard attempt cap600 includes consumed failures. Steering is legitimately skipped if gates fail. All fitted probes, TF-IDF baselines, random/permutation controls and steering vectors are stored/hashed before heldout.

Use the pinned isolated Python runtime with staged/bfcl and staged/model populated from the exact public revisions. Weights/cache/venv are excluded from publication. From this study workspace:
  python -B proposal/code/prepare.py --study-root .
  python -B proposal/code/run.py --study-root . --phase preflight
  python -B proposal/code/run.py --study-root . --phase dev-run
  python -B proposal/code/probes.py --study-root . --phase fit-dev
  python -B proposal/code/run.py --study-root . --phase dev-steer
  python -B proposal/code/gate.py --study-root . --write-hash-record
Root reviews exact source/config/split/fitted hashes and alone writes approvals/root-heldout-approval.json. Before that file exists, heldout is locked. Root also publishes the prospective freeze before main work.
  python -B proposal/code/run.py --study-root . --phase heldout
  python -B proposal/code/probes.py --study-root . --phase eval-heldout
  python -B proposal/code/analysis.py --study-root . --split DEV
  python -B proposal/code/analysis.py --study-root . --split HELDOUT

Raw tokens/strings, numerical diagnostics, activations and hashes are retained. Error cases stay in the scheduled bank; a numerical/runtime failure stops the run with its evidence. There are no invented human ratings, unverified human-contribution claims or official-BFCL-score claims. See ROOT_METHOD_AMENDMENTS.json and PROSPECTIVE_DESIGN.json.

TRANSPARENT V2 MEASUREMENT AMENDMENT
Development inspection showed that the V1 scorer wrongly marked a complete JSON name/arguments payload semantically incorrect merely because the closing tool tag was missing. V2 narrowly separates semantic reference-argument correctness from protocol format. Missing closing tags still fail strict format and invalid-output controls. Incomplete JSON, extra/unknown calls and bad arguments receive no new utility credit. Original V1 code/tag, all64original outputs/scores and original report are retained; all64outputs are re-scored identically into runs/dev/rescored_v2 with original file/line hashes. No generation, input, model, fitted probe or layer/C selection is repeated or changed. Historical derived scores are explicitly selected by every consumer; fresh V2 reruns use their own V2-labelled raw scores. See MEASUREMENT_AMENDMENT_V2.json and RESCORING_MANIFEST.json.
