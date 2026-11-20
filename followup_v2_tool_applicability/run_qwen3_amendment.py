"""Run the frozen pilot against the runtime-corrected Qwen3 server."""

import json
from pathlib import Path
import sys

from pilot import HERE, estimate, run_model, validate_bank


def main():
    bank = json.loads((HERE / "bank.json").read_text())
    validate_bank(bank)
    output = HERE / "results_qwen3_amended"
    output.mkdir(exist_ok=False)
    run_model("qwen3-4b", "http://127.0.0.1:8781", bank, output)
    rows = [json.loads(line) for line in (output / "qwen3-4b_raw.jsonl").read_text().splitlines()]
    results = estimate(rows)
    smoke = [row for row in rows if row["phase"] == "control"]
    admissible = (
        len(smoke) == 4
        and all("error" not in row and row["finish_reason"] != "length" for row in smoke)
        and all(row["score"]["canonical_correct_call"] for row in smoke if row["expected"] is not None)
    )
    results["amended_runtime_smoke_admissible"] = admissible
    (output / "summary.json").write_text(json.dumps({"qwen3-4b": results}, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    return int(not admissible or results["status"] != "complete")


if __name__ == "__main__":
    sys.exit(main())
