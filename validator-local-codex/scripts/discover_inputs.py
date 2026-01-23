#!/usr/bin/env python3
from __future__ import annotations
import os
import subprocess
from pathlib import Path

def _run(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode("utf-8", "replace").strip()

def _git_root(cwd: Path) -> Path | None:
    try:
        out = _run(["git", "-C", str(cwd), "rev-parse", "--show-toplevel"])
        return Path(out)
    except Exception:
        return None

def _find_submission(root: Path) -> Path | None:
    # Preferred: repo-root/submission
    cand = root / "submission"
    if cand.is_dir():
        return cand
    # Otherwise: search a few common locations relative to root
    for name in ("sub", "artifacts", "repo-artifacts", "platform-submission"):
        p = root / name
        if p.is_dir():
            return p
    # Otherwise: scan one level down for a folder containing Dockerfile.problem
    for p in root.iterdir():
        if p.is_dir() and (p / "Dockerfile.problem").is_file():
            return p
    return None

def _find_vlrun() -> Path | None:
    env = os.environ.get("VLRUN")
    if env:
        p = Path(env)
        if p.is_file():
            return p.resolve()
    home = Path.home()
    candidates = [
        home / "validator-local-codex" / "vlrun.py",
        home / "validator-local" / "vlrun.py",
        home / "validator-local-main" / "vlrun.py",
    ]
    for c in candidates:
        if c.is_file():
            return c.resolve()
    # best-effort: search a small set of dirs
    for base in (home, Path.cwd()):
        for p in base.glob("**/vlrun.py"):
            # avoid huge scans: stop early
            return p.resolve()
    return None

def _read_pinned_sha(sub_dir: Path) -> str | None:
    for fn in ("pinned_sha.txt", "PINNED_SHA.txt", "sha.txt"):
        p = sub_dir / fn
        if p.is_file():
            s = p.read_text(encoding="utf-8", errors="replace").strip()
            if s:
                return s.split()[0]
    return None

def main() -> None:
    cwd = Path.cwd()
    root = _git_root(cwd) or cwd
    sub = _find_submission(root)
    vlrun = _find_vlrun()
    print(f"REPO_DIR={root.resolve()}")
    if sub:
        print(f"SUB_DIR={sub.resolve()}")
    else:
        print("SUB_DIR=")
    if sub:
        sha = _read_pinned_sha(sub)
        print(f"PINNED_SHA={sha or ''}")
    else:
        print("PINNED_SHA=")
    print(f"VLRUN={vlrun or ''}")

if __name__ == "__main__":
    main()
