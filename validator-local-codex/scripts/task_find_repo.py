#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path


def _run(cmd: list[str], cwd: Path | None = None) -> str | None:
    try:
        out = subprocess.check_output(cmd, cwd=str(cwd) if cwd else None, stderr=subprocess.STDOUT)
        return out.decode("utf-8", "replace").strip()
    except Exception:
        return None


def _is_git_repo(p: Path) -> bool:
    return p.is_dir() and (p / ".git").exists()


def _has_commit(repo: Path, sha: str) -> bool:
    out = _run(["git", "-C", str(repo), "cat-file", "-e", f"{sha}^{{commit}}"])
    return out is not None


def _repo_name_from_url(url: str) -> str:
    # https://github.com/org/name(.git)? -> name
    url = url.strip()
    m = re.search(r"/([^/]+?)(?:\.git)?$", url)
    return m.group(1) if m else "repo"


def _git_root(cwd: Path) -> Path | None:
    out = _run(["git", "rev-parse", "--show-toplevel"], cwd=cwd)
    if out:
        p = Path(out)
        if p.is_dir():
            return p
    return None


def _pick_repo(cands: list[Path], pinned_sha: str | None) -> Path | None:
    for cand in cands:
        if not _is_git_repo(cand):
            continue
        if pinned_sha and not _has_commit(cand, pinned_sha):
            continue
        return cand.resolve()
    return None


def _clone_local(src: Path, dest: Path) -> bool:
    # Only clone if dest is missing or not a git repo.
    if dest.exists() and _is_git_repo(dest):
        return True
    if dest.exists() and not _is_git_repo(dest):
        # If user left a non-git folder here, do not destroy it.
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    out = _run(["git", "clone", str(src), str(dest)])
    return out is not None and _is_git_repo(dest)


def _try_clone_from_bundle(pinned_sha: str | None) -> Path | None:
    repo_url = os.environ.get("REPO_URL", "")
    name = _repo_name_from_url(repo_url) if repo_url else ""
    bundles_dir = Path("/workspace/validator-local-codex/bundles")
    # Compatibility: allow /workspace/bundles too.
    alt_bundles_dir = Path("/workspace/bundles")

    candidates: list[Path] = []

    def add_named(d: Path, base: str) -> None:
        if not base:
            return
        candidates.append(d / f"{base}.bundle")
        candidates.append(d / f"{base}.git")

    add_named(bundles_dir, name)
    add_named(alt_bundles_dir, name)
    add_named(Path("/workspace"), name)

    # If name is unknown/missing, fall back to "first bundle in bundles/"
    for d in (bundles_dir, alt_bundles_dir):
        if d.is_dir():
            for p in sorted(d.glob("*.bundle"))[:10]:
                candidates.append(p)
            for p in sorted(d.glob("*.git"))[:10]:
                candidates.append(p)

    src = None
    for p in candidates:
        if p.exists():
            src = p
            break
    if not src:
        return None

    dest = Path("/workspace/repo")
    if not _clone_local(src, dest):
        return None
    if pinned_sha and not _has_commit(dest, pinned_sha):
        return None
    return dest.resolve()


def find_repo() -> Path | None:
    pinned_sha = os.environ.get("PINNED_SHA")
    if os.environ.get("REPO_DIR"):
        p = Path(os.environ["REPO_DIR"])
        if _is_git_repo(p) and (not pinned_sha or _has_commit(p, pinned_sha)):
            return p.resolve()

    cwd = Path.cwd()
    root = _git_root(cwd)
    if root and _is_git_repo(root) and (not pinned_sha or _has_commit(root, pinned_sha)):
        return root.resolve()

    cands = [
        Path("/workspace/repo"),
        Path("/workspace/target-repo"),
        Path("/workspace/target_repo"),
        Path("/workspace/project"),
    ]
    picked = _pick_repo(cands, pinned_sha)
    if picked:
        return picked

    # As a last resort, try to materialize the repo from an offline bundle/mirror.
    return _try_clone_from_bundle(pinned_sha)


def main() -> None:
    p = find_repo()
    print(str(p) if p else "")


if __name__ == "__main__":
    main()
