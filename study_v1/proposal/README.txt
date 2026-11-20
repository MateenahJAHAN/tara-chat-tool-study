BFCL interface-package and applicability-probe study — prospective operator-amended version

This is a real public-data/open-weight experiment plan, using pinned Qwen2.5-1.5B-Instruct float32/MPS and BFCL v4 data. The actual Opus5.5 Max lead draft is preserved separately; independent input/scoring/runtime reviews led to pre-outcome operator amendments. No main inference result is claimed in this frozen plan.

Question: on this finite input-selected bank, compare false call attempts across a text catalogue and the model native tool template, then inspect cross-interface/source linear applicability probes with surface/random/permutation controls. This is an interface-package comparison and replication-style behavior baseline. The exploratory extension is controlled probe transfer and gated local steering, not general agent/biology safety. BFCL source categories supply labels and can confound probes; public benchmark contamination is unknown. No function/API is executed.

Input preparation passed: TRAIN64/class (128 rows), DEV16/class (32), HELDOUT32/class (64). Every selected row has a distinct connected-component group. C0 components are excluded from every main bank. All data/model revisions, hashes, IDs, prompts and fixed steering subsets are in config/manifests.

Expected maximum main generation schedule:192 baseline,128 dev steering,192 heldout steering =512. TRAIN uses prefill only. Two separate manual-runtime calibration repeats are permitted; hard attempt cap600 includes consumed failures. Steering is legitimately skipped if gates fail. All fitted probes, TF-IDF baselines, random/permutation controls and steering vectors are stored/hashed before heldout.

Use the pinned isolated Python runtime with staged/bfcl and staged/model populated from the exact public revisions. Weights/cache/venv are excluded from publication. From this study workspace, with CODE pointing to proposal/code:
  python -B CODE/prepare.py --study-root .
  python -B CODE/run.py --study-root . --phase preflight
  python -B CODE/run.py --study-root . --phase dev-run
  python -B CODE/probes.py --study-root . --phase fit-dev
  python -B CODE/run.py --study-root . --phase dev-steer
  python -B CODE/gate.py --study-root . --write-hash-record
Root reviews exact source/config/split/fitted hashes and alone writes approvals/root-heldout-approval.json. Before that file exists, heldout is locked. Root also publishes the prospective freeze before main work.
  python -B CODE/run.py --study-root . --phase heldout
  python -B CODE/probes.py --study-root . --phase eval-heldout
  python -B CODE/analysis.py --study-root . --split DEV
  python -B CODE/analysis.py --study-root . --split HELDOUT

Raw tokens/strings, numerical diagnostics, activations and hashes are retained. Error cases stay in the scheduled bank; a numerical/runtime failure stops the run with its evidence. There are no invented human ratings, unverified human-contribution claims or official-BFCL-score claims. See ROOT_METHOD_AMENDMENTS.json and PROSPECTIVE_DESIGN.json.
