"""Reconstruct the two frozen banks in the directory layout used at design time."""
import argparse
import hashlib
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, help="Directory for the three pinned downloads")
    args = parser.parse_args()
    external = Path(__file__).resolve().parent
    followup = external.parent
    repo = followup.parent
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        original = root / "research_advances_2026-11-21" / "TARA"
        bank = original / "independent_descriptions"
        bank.mkdir(parents=True)
        (root / "tara-chat-tool-study").symlink_to(repo, target_is_directory=True)
        (original / "bank_spec.json").symlink_to(followup / "bank_spec.json")
        (original / "transfer_01").symlink_to(followup / "transfer_01", target_is_directory=True)
        for name in ("prepare.py", "build_catalogue.py", "check_holdout.py"):
            shutil.copyfile(external / name, bank / name)
        cache = args.cache.resolve() if args.cache else root / "downloads"
        for arguments in (["prepare.py", "--cache", str(cache)], ["build_catalogue.py"], ["check_holdout.py"]):
            subprocess.run([sys.executable, *arguments], cwd=bank, check=True)
        for name in ("bank.jsonl", "sources.json", "catalogue_bank.jsonl", "holdout_check.json"):
            expected = hashlib.sha256((external / name).read_bytes()).hexdigest()
            actual = hashlib.sha256((bank / name).read_bytes()).hexdigest()
            if expected != actual:
                raise ValueError(f"{name} differed from the frozen copy")
            print(f"{actual}  {name}")


if __name__ == "__main__":
    main()
