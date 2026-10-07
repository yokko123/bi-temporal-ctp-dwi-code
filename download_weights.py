#!/usr/bin/env python3
"""Fetch the published model weights at the revisions pinned in models.yaml.

    python download_weights.py                     # every model
    python download_weights.py --model fe4_nnunet  # just one
    python download_weights.py --list              # show what is configured
    python download_weights.py --check             # verify an existing download

Each entry pins a Hugging Face commit sha rather than a branch, so repeated
downloads give byte-identical weights even after the model repo gains a new
release. After downloading, every file in `expect_files` is checked for exact
size, which catches a partial or interrupted transfer.

Destination, in order of precedence:
    --dest, then the environment variable named by `dest_env`, then
    `dest_default` resolved relative to this file.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

import yaml

HERE = pathlib.Path(__file__).resolve().parent
CONFIG = HERE / "models.yaml"


def load_config(path=CONFIG):
    if not path.exists():
        raise SystemExit(f"config not found: {path}")
    with open(path) as fh:
        return (yaml.safe_load(fh) or {}).get("models", {})


def resolve_dest(entry, override=None):
    if override:
        return pathlib.Path(override).expanduser().resolve()
    env = entry.get("dest_env")
    if env and os.environ.get(env):
        return pathlib.Path(os.environ[env]).expanduser().resolve()
    return (HERE / entry.get("dest_default", "./weights")).resolve()


def verify(dest, entry):
    """Return the list of problems; empty means the download is intact."""
    problems = []
    for rel, size in (entry.get("expect_files") or {}).items():
        p = dest / rel
        if not p.exists():
            problems.append(f"missing: {rel}")
        elif p.stat().st_size != size:
            problems.append(f"wrong size: {rel} ({p.stat().st_size} != {size})")
    return problems


def download(name, entry, override=None, check_only=False):
    dest = resolve_dest(entry, override)
    print(f"[{name}] {entry['repo_id']}@{entry['revision'][:12]}")
    print(f"  -> {dest}")

    if not check_only:
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            raise SystemExit("pip install huggingface_hub")
        dest.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id=entry["repo_id"],
            revision=entry["revision"],          # pinned, never a branch
            repo_type=entry.get("repo_type", "model"),
            local_dir=str(dest),
        )

    problems = verify(dest, entry)
    if problems:
        for p in problems:
            print(f"  FAIL  {p}")
        return False
    n = len(entry.get("expect_files") or {})
    print(f"  OK    {n} file(s) present at the expected size")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", help="only this entry from models.yaml")
    ap.add_argument("--dest", help="override the destination directory")
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--list", action="store_true", help="show configured models and exit")
    ap.add_argument("--check", action="store_true", help="verify without downloading")
    args = ap.parse_args()

    models = load_config(pathlib.Path(args.config))
    if not models:
        raise SystemExit("no models configured")

    if args.list:
        for name, e in models.items():
            total = sum((e.get("expect_files") or {}).values())
            print(f"  {name:14s} {e['repo_id']}@{e['revision'][:12]}  {total / 1e9:.2f} GB")
            print(f"  {'':14s} {' '.join((e.get('description') or '').split())}")
        return

    selected = {args.model: models[args.model]} if args.model else models
    if args.model and args.model not in models:
        raise SystemExit(f"unknown model {args.model!r}; have {list(models)}")

    ok = all(download(n, e, args.dest, args.check) for n, e in selected.items())
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
