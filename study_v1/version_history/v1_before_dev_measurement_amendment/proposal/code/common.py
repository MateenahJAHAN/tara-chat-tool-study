"""Shared helpers: config, hashing, JSONL IO, offline environment. Original code for BFCL_IFACE_PROBE_V1."""
import datetime
import hashlib
import json
import os
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent
PROPOSAL_DIR = CODE_DIR.parent
CONFIG_PATH = PROPOSAL_DIR / "config" / "study_config.json"
CONDITIONS = ("TEXT_CATALOGUE", "NATIVE_TOOLS")
QUERY_ONLY = "QUERY_ONLY_CONTROL"


def load_config(path=CONFIG_PATH):
    return json.loads(Path(path).read_text())


def now_utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def git_blob_sha1(path):
    data = Path(path).read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def canonical_json(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def tree_sha256(paths, base):
    """Hash of sorted relative paths plus file contents (code tree identity)."""
    h = hashlib.sha256()
    for p in sorted(Path(x) for x in paths):
        h.update(str(p.relative_to(base)).encode() + b"\0" + sha256_file(p).encode() + b"\n")
    return h.hexdigest()


def code_files():
    return sorted(CODE_DIR.glob("*.py"))


def read_jsonl(path):
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def append_jsonl(path, record):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def set_offline_env(study_root):
    for k, v in {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
                 "HF_HUB_DISABLE_TELEMETRY": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1",
                 "HF_HOME": str(Path(study_root) / ".cache" / "huggingface")}.items():
        os.environ[k] = v


def verify_staged(study_root, cfg, include_model=True):
    """Return a verification record; raise on any mismatch (stop rule)."""
    root = Path(study_root)
    rec = {"verified_at_utc": now_utc(), "data": {}, "model": {}}
    for name, exp in cfg["data"]["files"].items():
        p = root / "staged" / "bfcl" / name
        got = {"sha256": sha256_file(p), "git_blob_sha1": git_blob_sha1(p)}
        if got["sha256"] != exp["sha256"] or got["git_blob_sha1"] != exp["git_blob_sha1"]:
            raise RuntimeError(f"STOP: staged data mismatch {name}: {got}")
        rec["data"][name] = got
    if include_model:
        for name, exp in cfg["model"]["files_sha256"].items():
            got = sha256_file(root / "staged" / "model" / name)
            if got != exp:
                raise RuntimeError(f"STOP: staged model mismatch {name}")
            rec["model"][name] = got
    return rec
