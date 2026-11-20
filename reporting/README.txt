Reporting preparation (outside frozen inference code)

Actual explicitly authorized DEV V2 figures are now in figures_dev_v2:32 selected
items/components,64 paired baseline scored records; DEV-only selection-used
summaries. No heldout result has been read. Scripts still default to requirements-
only mode, which reads no results. Consult DEV_V2_REPORTING_STATUS.json and the
PUBLIC_REPORTING_ALLOWLIST.json for actual reviewed artifacts and exact hashes.

Working existing plotting interpreter:
Python3.12.11 (existing base interpreter), Matplotlib3.10.0, Agg backend.
No package installation or inference-environment change. Local executable paths
belong only in private authorization/launcher notes, never the public bundle.
The existing Homebrew Python3.11 has Matplotlib metadata but lacks dateutil and
cannot import pyplot; use the verified3.12 base instead.

After parent declares the exact phase completed, parent supplies an authorization
JSON with approved=true, completed=true, phase=DEV or HELDOUT, absolute main_root
(local authorization only), design_id, exact split_manifest_sha256,
and approved_reports mapping report-relative JSON paths to exact SHA256 hashes.
The default required report is reports/behaviour_dev.json orbehaviour_heldout.json.
An alternate reports-only filename can be named with behaviour_report. Authorize
reports/heldout_probe_results.json only after its actual approved completion.
HELDOUT also requires heldout_gate_approval_sha256 and heldout_complete_sha256,
binding actual root approval and runs/heldout/HELDOUT_COMPLETE.json.

For transparent versioned scorer amendments, authorization may supply scored_banks:
TEXT_CATALOGUE and NATIVE_TOOLS entries each containing path andsha256. These are
actual retained derived scored files, never a re-run. It may also supply a
scoring_version label. DEV authorization cannot read a path containing heldout.

Commands (replace every path with the actual approved artifact):
python3.12 -B reporting/build_reporting_context.py
python3.12 -B reporting/make_figures.py
These defaults print requirements only.

Explicit actual context mode:
python3.12 -B reporting/build_reporting_context.py --build --main-root MAIN_ROOT
  --phase DEV --authorization AUTHORIZATION_JSON
This reads the authorized completed behaviour report, frozen manifest, actual
phase-scored baseline files, and available selected/probe metadata. It verifies
complete paired UID banks and reported/raw counts, then writes only the new
reports/reporting_context_dev.json (exclusive; no overwrite). All boolean scorer
endpoints, stops/errors, attempt counts, validity diagnostics, row/group counts
and raw paired00/01/10/11 cells are retained. No new selection, tests or inference.

Explicit actual figure mode:
python3.12 -B reporting/make_figures.py --render --main-root MAIN_ROOT --phase DEV
  --authorization AUTHORIZATION_JSON --output-dir NEW_REPORTING_ARTIFACT_DIRECTORY
The output directory must be new and outside main_v1. This reads only authorized
reports JSON plus the phase context. It produces standalonePNG/PDF/SVG figures,
complete endpoint/status/paired-countCSV tables, a multipage endpointPDF and a
source-hashed figure audit. It never reads generation text or activations.

Rate intervals are copied from actual Wilson binomial-reference descriptors;
paired/probe intervals are copied from actual group-resampling reports. Labels
state that these are selected finite-bank exploratory descriptions, not
population-coverage guarantees or safety evidence. Missing/unrun metrics remain
explicit. No zero/status rows or disappointing outcomes are filtered for looks.
Steering displays actual selection/skipped/failed status and every reported
candidate/control; no success is imputed. DEV andHELDOUT are always separate.

Readiness validation uses syntax/default-mode/authorization/denominator tests
only. It creates no fake or placeholder research figures. Actual figure layout
still needs visual inspection after parent authorizes completed result reports.
No human authorship or personal course-completion inference is made.

Public privacy: contexts and figure audits identify the study by design_id plus
the exact split-manifestSHA256. They never publish main_root, executable or
machine-home paths. Private authorizations remain local-only. Shareable JSON is
checked for machine-home paths before writing. DEV and HELDOUT phase bindings
are checked independently; an absolute local authorization path is not copied.

Actual DEV layout QA: all final PNG layouts and all3 endpoint-PDF pages inspected.
The endpoint column was widened to prevent clipped labels. Small KL values use
significant-digit notation. A separate first-layout draft remains local-only.
The public allowlist excludes every authorization and Matplotlib/font cache.
No publication, model inference or raw-array export was performed.
