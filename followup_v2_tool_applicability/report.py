#!/usr/bin/env python3
"""Verify saved TARA runs and render the follow-up report and figures."""

from collections import Counter
import hashlib
import json
import subprocess

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pilot import HERE, MODES, build_cases, estimate, request_body, score


DISPLAY = {"qwen2.5-1.5b": "Qwen2.5 1.5B", "qwen3.5-4b": "Qwen3.5 4B"}
COLORS = {"text": "#38587A", "native": "#118175"}
METRICS = [
    ("false_calls", "False calls", "Out-of-scope request"),
    ("semantic_correct_calls", "Correct call", "In-scope request"),
    ("canonical_correct_calls", "Canonical correct call", "In-scope request"),
    ("scope_pairs_passed", "Scope pair passed", "Exact call and explicit NO_CALL"),
]


def load_json(path):
    return json.loads(path.read_text())


def record_digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_run(path, bank, diagnostic=False):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    expected = {row["id"]: row for row in build_cases(bank, True) + build_cases(bank)}
    assert (4 <= len(rows) <= 84 if diagnostic else len(rows) == 84), f"{path}: unexpected JSONL length"
    seen = set()
    for row in rows:
        key = (row["id"], row["mode"])
        assert key not in seen and row["mode"] in MODES
        seen.add(key)
        case = expected[row["id"]]
        assert all(row[field] == case[field] for field in case)
        assert row["request"] == request_body(row["request"]["model"], bank["tools"], case["query"], row["mode"])
        if "error" not in row:
            first = row["response"]["choices"][0]
            assert row["finish_reason"] == first["finish_reason"]
            assert row["score"] == score(first["message"], row["mode"], row["expected"])
    return rows


def breakdown(rows):
    controls = [row for row in rows if row["phase"] == "control"]
    main = [row for row in rows if row["phase"] == "heldout"]
    valid = [row for row in main if "response" in row]
    variants = Counter()
    for row in valid:
        if row["expected"] is not None or row["score"]["call_attempt"] or row["score"]["exact_abstain"]:
            continue
        output = (row["response"]["choices"][0]["message"].get("content") or "").strip()
        if output.replace("_", "").replace(" ", "").casefold() == "nocall":
            variants[row["mode"] + ":" + output] += 1
    by_family = {}
    for row in valid:
        if row["expected"] is not None:
            continue
        counts = by_family.setdefault(row["family"], {mode: {"n": 0, "false_calls": 0} for mode in MODES})
        counts[row["mode"]]["n"] += 1
        counts[row["mode"]]["false_calls"] += int(row["score"]["call_attempt"])
    return {
        "requests": len(rows),
        "received_heldout": len(valid),
        "error_heldout": len(main) - len(valid),
        "heldout_length_caps": sum(row["finish_reason"] == "length" for row in valid),
        "control_length_caps": sum(row.get("finish_reason") == "length" for row in controls),
        "control_exact_no_call": [
            row["mode"] for row in controls if row["expected"] is None and row.get("score", {}).get("exact_abstain")
        ],
        "heldout_out_of_scope_abstention_variants": dict(variants),
        "heldout_false_calls_by_family": by_family,
    }


def plot_scores(results, target):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "text.color": "#14263A"})
    fig, axes = plt.subplots(2, 2, figsize=(11.7, 7.6), sharey=True)
    bg = "#F3F7FA"
    fig.patch.set_facecolor(bg)
    for ax, (field, title, subtitle) in zip(axes.flat, METRICS):
        ax.set_facecolor("#FFFFFF")
        ax.spines[["right", "top", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#C8D8E8")
        ax.grid(axis="y", color="#E4EDF4", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_title(title + "\n" + subtitle, loc="left", pad=13, fontsize=11, linespacing=1.4)
        for idx, mode in enumerate(MODES):
            x = [index + (-0.16, 0.16)[idx] for index in range(len(results))]
            rows = [result["conditions"][mode] for result in results.values()]
            heights = [100 * row[field] / row["n_pairs"] for row in rows]
            ax.bar(x, heights, width=0.29, color=COLORS[mode], label=f"{mode.capitalize()} interface")
            for loc, height, row in zip(x, heights, rows):
                ax.text(loc, height + 3, f'{row[field]}/{row["n_pairs"]}', ha="center", fontsize=9)
        ax.set_xlim(-.58, len(results)-.42)
        ax.set_xticks(range(len(results)), [DISPLAY[name] for name in results])
        ax.set_ylim(0, 116)
        ax.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
        ax.tick_params(axis="both", labelsize=10, length=0)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(.96, .881), ncol=2, frameon=False, fontsize=10)
    fig.suptitle("Tool applicability on 20 matched scope pairs", x=.079, y=.962, ha="left", fontsize=19, weight="bold")
    fig.text(.079, .918, "Real local generations on a fixed synthetic bank. No functions executed.", fontsize=10)
    fig.subplots_adjust(left=.085, right=.962, top=.812, bottom=.09, wspace=.21, hspace=.55)
    fig.text(.085, .027, "Counts are per model and interface. Canonical formatting is stricter than semantic call correctness.", fontsize=9, color="#445D73")
    fig.savefig(target, dpi=190, facecolor=bg, bbox_inches="tight")
    plt.close(fig)


def report_text(results, audits):
    lines = [
        "# TARA follow-up: matched tool applicability",
        "",
        "We tested 20 mock scope pairs on the original Qwen2.5-1.5B-Instruct checkpoint and a local Qwen3.5-4B checkpoint. Each pair changes one scope word while keeping the user request and visible tool catalogue comparable. A pair passes when its applicable prompt produces the exact call and its out-of-scope counterpart produces explicit NO_CALL. The experiment executes no tools. Claude Fable 5.2 reviewed the proposed design before inference.",
        "",
        "## Results",
        "",
        "| Model | Interface | False calls | Exact abstentions | Correct calls | Canonical correct calls | Scope pairs passed | Length-capped outputs |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model, report in results.items():
        for mode, values in report["conditions"].items():
            n = values["n_pairs"]
            cells = [f"{values[key]}/{n}" for key in (
                "false_calls", "exact_abstentions", "semantic_correct_calls", "canonical_correct_calls", "scope_pairs_passed"
            )]
            cells.append(f"{values['generation_caps']}/{2*n}")
            lines.append("| " + " | ".join([DISPLAY[model], mode, *cells]) + " |")
    lines += ["", "![Four behavioural endpoints with exact denominators](figures/endpoint_counts.png)", "", "### Paired false-call contrast", ""]
    for model, report in results.items():
        counts = report["paired_false_calls"]
        diff = report["difference_native_minus_text_pp"]
        lo, hi = report["bootstrap_95_descriptor_pp"]
        sign = report["exact_two_sided_discordance_sign_p"]
        lines.append(
            f"- **{DISPLAY[model]}.** Native minus text was {diff:+g} percentage points. "
            f"Fixed-family bootstrap descriptor {lo:+g} to {hi:+g} points, exact discordance sign test "
            f"p={sign:.4g}. Discordant pairs were {counts['native_only']} native-only and "
            f"{counts['text_only']} text-only, with {counts['both']} both and {counts['neither']} neither."
        )
    lines.extend([
        "",
        "| Tool family | Qwen2.5 text/native false calls | Qwen3.5 text/native false calls |",
        "| --- | --- | --- |",
    ])
    for family in sorted(audits["qwen2.5-1.5b"]["heldout_false_calls_by_family"]):
        cells = []
        for model in results:
            rates = audits[model]["heldout_false_calls_by_family"][family]
            cells.append(" vs ".join(f'{rates[mode]["false_calls"]}/{rates[mode]["n"]}' for mode in MODES))
        lines.append("| " + " | ".join([family, *cells]) + " |")
    legacy = audits["qwen3_legacy_runtime"]
    amended = audits["qwen3_amended_runtime"]
    lines += [
        "",
        "## Runtime checks and scope",
        "",
        f"- Qwen3-4B put scratchpad text in its final-answer field despite the requested non-thinking setting. All {legacy['control_length_caps']} original and {amended['control_length_caps']} amended smoke responses hit the 96-token cap without an applicable call. Both excluded attempts are archived. The original file contains {legacy['received_heldout']} received held-out responses and {legacy['error_heldout']} interruption errors. The amended file contains {amended['received_heldout']} received held-out responses before stopping.",
        "- The same frozen request bank, catalogue, scorer, greedy decoding and token cap apply to Qwen2.5 and Qwen3.5. Qwen2.5 uses its original Transformers/MPS stack and Qwen3.5 uses llama.cpp with quantized weights. Within each model, the tool format is the planned contrast. Across models, checkpoints, templates, precision and runtime also differ.",
    ]
    for model in results:
        variants = audits[model]["heldout_out_of_scope_abstention_variants"]
        if variants:
            description = ", ".join(f"{count} {label.split(':', 1)[1]} in {label.split(':', 1)[0]}" for label, count in sorted(variants.items()))
            lines.append(f"- **{DISPLAY[model]} abstention variants.** {description}. The exact abstention and pair metrics do not credit spelling variants. The false-call metric counts recognizable tool attempts separately.")
    lines += [
        "- The 160 matched holdout generations are all present across the two comparison runs. The records include exact request bodies, raw response messages, runtime versions, templates, and weight identifiers. The figure and table can be regenerated from those records.",
        "- These results describe 20 constructed pairs with five repeated tool families. The design controls wording within each pair. Claims about model families, naturally occurring requests, agent safety, or internal mechanisms would require separate validation.",
        "",
        "## Design",
        "",
        "![Matched scope-pair experimental diagram](figures/study_design.svg)",
        "",
        "[Frozen protocol](PROTOCOL.md) · [Exact bank and schemas](bank.json) · [Qwen3 amendment](AMENDMENT_QWEN3.md) · [Qwen2.5 amendment](AMENDMENT_QWEN25.md) · [Qwen3.5 and legacy Qwen3 records](results/) · [Qwen2.5 records](results_qwen25/) · [Amended Qwen3 diagnostic records](results_qwen3_amended/) · [Machine-readable verification](verification.json)",
    ]
    return "\n".join(lines) + "\n"


def render_design(path):
    diagram = r'''digraph matched_pilot {
        graph [rankdir=TB, bgcolor="#F3F7FA", pad=".23", nodesep=".34", ranksep=".38", fontname="Helvetica"];
        node [shape=box, style="rounded,filled", margin=".18,.10", fillcolor="white", color="#A6B8C9", fontname="Helvetica", fontsize=12, fontcolor="#14263A"];
        edge [color="#65839D", arrowsize=.65];
        bank [label="20 authored scope pairs\nFive tool families"];
        relevant [label="Whole request in scope\nGold: one exact call"];
        irrelevant [label="One scope word changes\nGold: NO_CALL"];
        text [label="Text JSON catalogue", fillcolor="#E1EBF6", color="#38587A"];
        native [label="Native tool declarations", fillcolor="#DDF3EF", color="#118175"];
        model [label="Two local checkpoints\nQwen2.5 1.5B and Qwen3.5 4B"];
        outcomes [label="False calls, exact abstention, correct calls\nCanonical format and paired performance"];
        bank -> relevant;
        bank -> irrelevant;
        {rank=same; relevant; irrelevant;}
        relevant -> text;
        relevant -> native;
        irrelevant -> text;
        irrelevant -> native;
        {rank=same; text; native;}
        text -> model;
        native -> model;
        model -> outcomes;
    }'''
    subprocess.run(["dot", "-Tsvg", "-o", str(path)], input=diagram, text=True, check=True)


def main():
    bank = load_json(HERE / "bank.json")
    raw_paths = {
        "qwen2.5-1.5b": HERE / "results_qwen25/qwen2.5-1.5b_raw.jsonl",
        "qwen3.5-4b": HERE / "results/qwen3.5-4b_raw.jsonl",
        "qwen3_legacy_runtime": HERE / "results/qwen3-4b_raw.jsonl",
        "qwen3_amended_runtime": HERE / "results_qwen3_amended/qwen3-4b_raw.jsonl",
    }
    raw = {name: verify_run(path, bank, diagnostic=name not in DISPLAY) for name, path in raw_paths.items()}
    results = {name: estimate(raw[name]) for name in DISPLAY}
    assert all(value["status"] == "complete" and value["records"] == 80 for value in results.values())
    qc = load_json(HERE / "results_qwen25/summary.json")["qwen2.5-1.5b"]
    assert qc.pop("runtime_smoke_admissible")
    assert qc == results["qwen2.5-1.5b"]
    assert load_json(HERE / "results/summary.json")["qwen3.5-4b"] == results["qwen3.5-4b"]
    audits = {name: breakdown(rows) for name, rows in raw.items()}
    verification = {
        "bank_sha256": record_digest(HERE / "bank.json"),
        "protocol_sha256": record_digest(HERE / "PROTOCOL.md"),
        "raw_records_checked": sum(len(rows) for rows in raw.values()),
        "received_scores_recomputed": sum("score" in row for rows in raw.values() for row in rows),
        "sha256": {str(path.relative_to(HERE)): record_digest(path) for path in raw_paths.values()},
        "results": results,
        "runtime_audit": audits,
    }
    figures = HERE / "figures"
    figures.mkdir(exist_ok=True)
    plot_scores(results, figures / "endpoint_counts.png")
    render_design(figures / "study_design.svg")
    (HERE / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    (HERE / "RESULTS.md").write_text(report_text(results, audits))
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
