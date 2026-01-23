#!/usr/bin/env python3
"""
validator-local: single-file FastAPI service for validating platform-style Python problem submissions.

Notes:
- This file is intentionally self-contained and avoids network calls (no HTTP fetches).
- All persistent artifacts live under ./data/jobs/<job_id>/ relative to this file.
"""
from __future__ import annotations
import random

import json
import os
import re
import math
import ast
import shutil
import subprocess
import threading
import time
import uuid
import hashlib
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Literal, Optional, Tuple

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field, AliasChoices
from pydantic.config import ConfigDict


# -----------------------------
# Paths / constants
# -----------------------------

APP_ROOT = Path(__file__).resolve().parent
DATA_DIR = APP_ROOT / "data"
JOBS_DIR = DATA_DIR / "jobs"

DEFAULT_WAIT_TIMEOUT_SEC = 60 * 30  # 30m


# -----------------------------
# Helpers: time / JSON / IO
# -----------------------------

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _safe_name(name: str) -> str:
    # Prevent traversal; keep it simple and strict.
    if not name or "/" in name or "\\" in name or name.startswith(".") or ".." in name:
        raise ValueError("invalid artifact name")
    if "\x00" in name:
        raise ValueError("invalid artifact name")
    return name


def _read_text_file(path: Path, *, encoding: str = "utf-8") -> str:
    try:
        return path.read_text(encoding=encoding)
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail=f"File not found: {path}") from None
    except UnicodeDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode file as {encoding}: {path} ({e})") from None


def _write_text_file(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # force LF
    path.write_text(text.replace("\r\n", "\n"), encoding="utf-8", newline="\n")


def _read_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail=f"File not found: {path}") from None


def _json_dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    tmp.replace(path)


def _json_load(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="job not found") from None
    except json.JSONDecodeError:
        # Never 500 on status endpoints.
        return {"status": "failed", "reason": f"corrupt status file: {path.name}", "crash_log": "crash.log"}


def _append_log(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(line.replace("\r\n", "\n"))
        if not line.endswith("\n"):
            f.write("\n")
        f.flush()


def _log(job_dir: Path, which: str, msg: str) -> None:
    ts = _utc_now_iso()
    _append_log(job_dir / which, f"[{ts}] {msg}")


# -----------------------------
# Models (Pydantic v2)
# -----------------------------

class PolicyOptions(BaseModel):
    enforce_description_coverage: bool = False
    ai_curve_center: float = 88.0
    ai_curve_slope: float = 7.0
    model_config = ConfigDict(extra="ignore")

    enforce_single_new_test_file: bool = True
    enforce_new_test_suffix_problem: bool = True  # tests/test_*_problem.py
    enforce_ascii_lf: bool = True
    enforce_no_test_leakage_in_description: bool = True
    enforce_minimal_test_sh: bool = True
    forbid_tail_piping_in_test_sh: bool = True
    forbid_solution_touch_tests: bool = True
    forbid_solution_touch_dockerfiles: bool = True


    # Description lint / AI difficulty heuristics (local, deterministic; no network)
    enforce_description_present: bool = False
    enforce_no_solution_hints_in_description: bool = True
    description_min_requirements: int = 6
    enforce_description_min_requirements: bool = False  # set True to FAIL when too vague

    ai_heuristics_enabled: bool = True
    ai_target_pass_rate_min: float = 0.10
    ai_target_pass_rate_max: float = 0.35
    ai_enforce_target_band: bool = True  # if True: fail when outside band
    ai_sim_trials: int = 5
    ai_sim_passes_max: int = 1


class StaticInlineRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    dockerfile_text: str = Field(
        validation_alias=AliasChoices("dockerfile_text", "dockerfile", "dockerfile_contents"),
    )
    test_patch_text: str = Field(
        validation_alias=AliasChoices("test_patch_text", "test_patch", "test_patch_contents"),
    )
    solution_patch_text: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("solution_patch_text", "solution_patch", "solution_patch_contents"),
    )
    description_text: Optional[str] = Field(
        default=None,
        validation_alias=AliasChoices("description_text", "description", "problem_description"),
    )
    policy: PolicyOptions = Field(default_factory=PolicyOptions)


class StaticFromFilesRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    dockerfile_path: str
    test_patch_path: str
    solution_patch_path: Optional[str] = None
    description_path: Optional[str] = None
    policy: PolicyOptions = Field(default_factory=PolicyOptions)


class JobFromFilesRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    repo_path: str
    sha: str
    dockerfile_path: str
    test_patch_path: str
    solution_patch_path: str
    description_path: Optional[str] = None

    # Backwards compatible: optional wait controls.
    wait: Optional[bool] = None
    wait_timeout_sec: Optional[float] = None
    policy: PolicyOptions = Field(default_factory=PolicyOptions)


class JobInlineRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    repo_path: str
    sha: str
    dockerfile_text: str = Field(
        validation_alias=AliasChoices("dockerfile_text", "dockerfile", "dockerfile_contents"),
    )
    test_patch_text: str = Field(
        validation_alias=AliasChoices("test_patch_text", "test_patch", "test_patch_contents"),
    )
    solution_patch_text: str = Field(
        validation_alias=AliasChoices("solution_patch_text", "solution_patch", "solution_patch_contents"),
    )

    wait: Optional[bool] = None
    wait_timeout_sec: Optional[float] = None
    policy: PolicyOptions = Field(default_factory=PolicyOptions)


# -----------------------------
# Static check machinery
# -----------------------------

@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    severity: Literal["PASS", "FAIL", "WARN", "INFO"]
    description: str
    message: str
    details: Dict[str, Any]


def _check(name: str, *, passed: bool, description: str, message: str, details: Dict[str, Any], severity_if_fail: str = "FAIL") -> Check:
    if passed:
        sev: Literal["PASS", "FAIL", "WARN", "INFO"] = "PASS"
    else:
        sev = severity_if_fail  # type: ignore[assignment]
        if sev not in {"FAIL", "WARN", "INFO"}:
            sev = "FAIL"
    return Check(name=name, passed=passed, severity=sev, description=description, message=message, details=details)


def _check_warn(name: str, *, passed: bool, description: str, message: str, details: Dict[str, Any]) -> Check:
    return _check(name, passed=passed, description=description, message=message, details=details, severity_if_fail="WARN")


_DIFF_GIT_RE = re.compile(r"^diff --git a/(.+?) b/(.+?)\s*$")
_PLUS_FILE_RE = re.compile(r"^\+\+\+ b/(.+?)\s*$")
_MINUS_FILE_RE = re.compile(r"^--- a/(.+?)\s*$")
_NEW_FILE_MODE_RE = re.compile(r"^new file mode \d+\s*$")
_DEL_FILE_MODE_RE = re.compile(r"^deleted file mode \d+\s*$")


def _parse_changed_files_unified_diff(patch_text: str) -> Dict[str, Dict[str, Any]]:
    """
    Best-effort parser for unified diffs. Returns mapping:
        path -> {"a_path": str|None, "b_path": str|None, "is_new": bool, "is_deleted": bool}
    """
    files: Dict[str, Dict[str, Any]] = {}
    cur: Optional[Dict[str, Any]] = None

    lines = patch_text.splitlines()
    for ln in lines:
        m = _DIFF_GIT_RE.match(ln)
        if m:
            a_path, b_path = m.group(1), m.group(2)
            cur = {"a_path": a_path, "b_path": b_path, "is_new": False, "is_deleted": False}
            # Use b_path as key by default.
            files[b_path] = cur
            continue
        if cur is None:
            continue
        if _NEW_FILE_MODE_RE.match(ln):
            cur["is_new"] = True
            continue
        if _DEL_FILE_MODE_RE.match(ln):
            cur["is_deleted"] = True
            continue
        # Some diffs omit 'new file mode' and use /dev/null.
        if ln.startswith("--- /dev/null"):
            cur["is_new"] = True
            continue
        if ln.startswith("+++ /dev/null"):
            cur["is_deleted"] = True
            continue

    return files


def _scan_ascii_lf(name: str, content: str, *, enforce: bool) -> List[str]:
    issues: List[str] = []
    b = content.encode("utf-8", errors="strict")
    if b"\r\n" in b:
        issues.append(f"{name}: CRLF line endings detected (must be LF-only)")
    if enforce:
        try:
            content.encode("ascii")
        except UnicodeEncodeError as e:
            issues.append(f"{name}: non-ASCII characters detected ({e})")
    return issues


def _dockerfile_checks(dockerfile_text: str) -> List[Check]:
    checks: List[Check] = []
    desc = "Dockerfile must follow the local platform dev-shell constraints."
    lines = dockerfile_text.replace("\r\n", "\n").splitlines()
    first_nonempty = ""
    for ln in lines:
        if ln.strip():
            first_nonempty = ln.strip()
            break

    checks.append(
        _check(
            "Dockerfile base image",
            passed=(first_nonempty.startswith("FROM public.ecr.aws/x8v8d7g8/mars-base:latest")),
            description=desc,
            message=("OK" if first_nonempty else "Missing FROM line") if first_nonempty else "Missing FROM line",
            details={"first_nonempty_line": first_nonempty},
        )
    )

    # pip upgrade ban (strict)
    pip_upgrade = bool(re.search(r"\bpip\s+install\b.*\b(--upgrade|-U)\b.*\bpip\b", dockerfile_text))
    checks.append(
        _check(
            "Dockerfile must not upgrade pip",
            passed=(not pip_upgrade),
            description="Do not run 'pip install -U pip' / '--upgrade pip' during build (common validator issue).",
            message="OK" if not pip_upgrade else "Found pip upgrade invocation",
            details={"matched": pip_upgrade},
        )
    )

    # no tests during build
    # Heuristic: flag RUN layers that *execute* pytest/test.sh, but do not flag installing pytest as a dependency.
    run_pytest = False
    matched_run_line = None
    for ln in lines:
        if not ln.strip().upper().startswith("RUN "):
            continue
        cmd = ln.strip()[3:].strip()

        # Allow dependency installation (common false-positive: "pip install pytest").
        if re.search(r"^\s*(python\s+-m\s+)?pip\d*\s+install\b", cmd):
            continue
        if re.search(r"^\s*python\s+-m\s+pip\s+install\b", cmd):
            continue

        # Flag execution of pytest or test.sh in shell form (including in chains).
        if re.search(
            r"(^|[;&|]\s*|&&\s*)("
            r"pytest(\s|$)|"
            r"python\s+-m\s+pytest(\s|$)|"
            r"\.\/test\.sh(\s|$)|"
            r"bash\s+(\./)?test\.sh(\s|$)"
            r")",
            cmd,
            flags=re.IGNORECASE,
        ):
            run_pytest = True
            matched_run_line = ln.strip()
            break

    checks.append(
        _check(
            "Dockerfile must not run tests during build",
            passed=(not run_pytest),
            description="Dockerfile should build a dev shell only; do not execute tests during image build.",
            message="OK" if not run_pytest else "Found test execution in RUN layer",
            details={"matched": run_pytest, "matched_line": matched_run_line},
        )
    )

    # ends with bash
    ends_with_bash = any(re.match(r'^\s*CMD\s+\["/bin/bash"\]\s*$', ln) for ln in lines)
    checks.append(
        _check_warn(
            "Dockerfile ends with /bin/bash",
            passed=ends_with_bash,
            description="Usually expected for local dev-shell Dockerfiles.",
            message="OK" if ends_with_bash else 'Missing: CMD ["/bin/bash"]',
            details={"has_cmd_bash": ends_with_bash},
        )
    )

    return checks


def _extract_file_diff_block(patch_text: str, target_path: str) -> Optional[str]:
    """
    Return the unified-diff block (as text) for the given path, if present.
    This is useful for heuristics when we cannot reconstruct full file content.
    """
    marker = f"diff --git a/{target_path} b/{target_path}"
    i = patch_text.find(marker)
    if i == -1:
        return None
    j = patch_text.find("\ndiff --git ", i + 1)
    if j == -1:
        return patch_text[i:]
    return patch_text[i:j]


def _extract_new_file_content_from_patch(patch_text: str, target_path: str) -> Optional[str]:
    """
    If patch adds a new file at target_path, extract its full content from the diff.
    Returns None if not confidently extractable.
    """
    # Identify the file block.
    lines = patch_text.splitlines(True)
    start = None
    in_target = False
    is_new = False

    for i, ln in enumerate(lines):
        if ln.startswith(f"diff --git a/{target_path} b/{target_path}"):
            start = i
            in_target = True
            is_new = False
            continue
        if in_target and ln.startswith("diff --git "):
            break
        if in_target and ln.startswith("--- /dev/null"):
            is_new = True
        if in_target and ln.startswith("new file mode"):
            is_new = True

    if not in_target or start is None or not is_new:
        return None

    # Extract all added lines from hunks.
    content_lines: List[str] = []
    in_hunk = False
    for ln in lines[start:]:
        if ln.startswith("diff --git ") and content_lines:
            break
        if ln.startswith("@@"):
            in_hunk = True
            continue
        if not in_hunk:
            continue
        if ln.startswith("+") and not ln.startswith("+++"):
            content_lines.append(ln[1:])
        elif ln.startswith("\\ No newline at end of file"):
            continue
        elif ln.startswith("-") and not ln.startswith("---"):
            # ignore removals
            continue
        else:
            # context lines start with space
            if ln.startswith(" "):
                content_lines.append(ln[1:])

    return "".join(content_lines)


_LEAKY_DESC_PATTERNS = [
    r"\bpytest\b",
    r"\btest\.sh\b",
    r"\btests/\b",
    r"\btest_\w+\b",
    r"\bconftest\.py\b",
]



_AI_SOLUTION_HINT_PATTERNS = [
    r"\bjust\s+do\b",
    r"\bsimply\b",
    r"\beasy\b",
    r"\bone[- ]liner\b",
    r"\badd\s+(?:a|an|the)\s+if\b",
    r"\bchange\s+this\s+line\b",
    r"\bfix\s+by\b",
    r"\buse\s+regex\b",
    r"\bwrap\s+in\s+try\b",
    r"\bcatch\s+the\s+exception\b",
    r"\bcall\s+this\s+function\b",
    r"\bupdate\s+the\s+return\b",
]

def _description_lint(text: str) -> Dict[str, Any]:
    # Goal: requirements-only, self-contained, fair. Not "how to implement".
    hits: List[str] = []
    for pat in _AI_SOLUTION_HINT_PATTERNS:
        if re.search(pat, text, flags=re.IGNORECASE):
            hits.append(pat)

    req_words = re.findall(r"\b(must\s+not|must|should\s+not|should)\b", text, flags=re.IGNORECASE)
    file_mentions = re.findall(r"\b[\w./-]+\.(?:py|toml|sh)\b", text)
    dotted = re.findall(r"\b[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*\b", text)
    callish = re.findall(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\(", text)

    suggestions: List[str] = []
    if hits:
        suggestions.append("Remove solution-hint phrasing; keep it requirements-only (what, not how).")
    if len(file_mentions) > 3 or len(dotted) > 6 or len(callish) > 8:
        suggestions.append("Too many internal identifiers; rewrite behaviorally so it's fair/self-contained.")
    if len(req_words) < 3:
        suggestions.append("Consider stating more explicit requirements (must/must not) to define the win condition.")

    return {
        "solution_hint_patterns": hits,
        "requirements_word_count": len(req_words),
        "file_mentions": file_mentions[:20],
        "dotted_identifiers": dotted[:30],
        "call_like_tokens": callish[:30],
        "suggestions": suggestions,
    }

def _infer_new_test_path_from_patch(patch_text: str) -> Optional[str]:
    # diff --git a/tests/foo.py b/tests/foo.py + "new file mode"
    for m in re.finditer(r"^diff --git a/(tests/[^\s]+\.py)\s+b/\1\s*$", patch_text, flags=re.MULTILINE):
        path = m.group(1)
        block_start = m.start()
        block_end = patch_text.find("\ndiff --git ", block_start + 1)
        if block_end == -1:
            block_end = len(patch_text)
        block = patch_text[block_start:block_end]
        if re.search(r"^new file mode\s+\d+", block, flags=re.MULTILINE):
            return path
    return None

def _simple_test_metrics(test_text: str) -> Dict[str, Any]:
    return {
        "num_tests": len(re.findall(r"^\s*def\s+test_", test_text, flags=re.MULTILINE)),
        "asserts": len(re.findall(r"\bassert\b", test_text)),
        "raises_ctx": len(re.findall(r"pytest\.raises\(", test_text)),
        "parametrize": len(re.findall(r"parametrize\b", test_text)),
        "hypothesis": bool(re.search(r"\bhypothesis\b|\bgiven\b", test_text)),
        "bool_traps": len(re.findall(r"def\s+__bool__\b", test_text)),
        "monkeypatch": bool(re.search(r"\bmonkeypatch\b", test_text)),
    }


def _compute_description_coverage(*, description_text: str | None, test_patch_text: str, policy: Any) -> Dict[str, Any]:
    """Heuristic fairness guard: flag likely-undocumented requirements.
    No LLM runs; compares keywords implied by tests vs the description.
    """
    if not description_text:
        return {"passed": True, "required_concepts": [], "covered_concepts": [], "missing_concepts": [], "note": "no description provided"}
    desc = description_text.lower()
    tests = (test_patch_text or "").lower()
    concepts = [
        {"name": "default_inherits_console_highlighter", "test_any": ["console.highlighter", "make_console("], "desc_any": ["inherit", "console highlighter", "console.highlighter"]},
        {"name": "tracks_runtime_console_highlighter_changes", "test_any": ["console.highlighter =", "toggle", "tracks_console_highlighter"], "desc_any": ["track", "runtime", "changes", "reflect"]},
        {"name": "tracks_handler_console_reassignment", "test_any": ["handler.console =", "reassignment"], "desc_any": ["handler.console", "reassign", "reassignment"]},
        {"name": "explicit_handler_highlighter_override_is_stable", "test_any": ["highlighter=", "override"], "desc_any": ["explicit", "override", "must not change", "stable", "remains"]},
        {"name": "per_record_highlighter_override", "test_any": ["record.highlighter", "logrecord.highlighter"], "desc_any": ["record.highlighter", "logrecord.highlighter", "per-record"]},
        {"name": "none_disables_highlighting", "test_any": ["none_disables", "highlighter = none"], "desc_any": ["none disables"]},
        {"name": "nullhighlighter_behavior", "test_any": ["nullhighlighter"], "desc_any": ["nullhighlighter"]},
        {"name": "reprhighlighter_fallback_when_none", "test_any": ["reprhighlighter"], "desc_any": ["reprhighlighter", "fallback", "fall back"]},
        {"name": "falsy_highlighters_are_valid", "test_any": ["explodingboolhighlighter", "__bool__", "truthiness"], "desc_any": ["falsy", "truthiness", "__bool__", "must not"]},
        {"name": "no_ansi_when_highlighting_disabled", "test_any": ["ansi", "\\x1b["], "desc_any": ["ansi", "no ansi", "plain text", "no styling"]},
        {"name": "keyword_highlighting_only_when_enabled", "test_any": ["keyword", "keywords"], "desc_any": ["keyword", "keywords", "only when", "enabled"]},
    ]
    required, covered, missing = [], [], []
    for c in concepts:
        if any(tok in tests for tok in c["test_any"]):
            required.append(c["name"])
            if any(tok in desc for tok in c["desc_any"]):
                covered.append(c["name"])
            else:
                missing.append(c["name"])
    return {"passed": len(missing) == 0, "required_concepts": required, "covered_concepts": covered, "missing_concepts": missing}


def _hintiness_diagnose(description_text, test_patch_text):
    """
    Deterministic, offline heuristic: detect when the problem is "procedural/spec-mirrored",
    which tends to be AI-easy even when tests are large.
    Returns: {"points": int, "spec_overlap": float, "penalty": float, "signals": [str]}
    """
    import re

    desc = (description_text or "")
    test = (test_patch_text or "")

    # cheap token sets for overlap
    stop = {
        "the","and","or","for","with","that","this","then","else","when","if","only",
        "must","should","will","from","into","over","under","true","false","none",
        "return","raise","raises","error","errors","case","cases","test","tests",
        "file","files","path","paths","value","values","string","strings","name","names",
        "function","class","module","package","resource","resources",
    }

    def words(text):
        ws = re.findall(r"[a-zA-Z_]{4,}", text.lower())
        return {w for w in ws if w not in stop}

    dkw = words(desc)
    tkw = words(test)
    overlap = (len(dkw & tkw) / max(1, len(dkw))) if dkw else 0.0

    points = 0
    signals = []

    # Procedural structure signals
    backticks = desc.count("`")
    if backticks >= 6:
        points += min(30, (backticks // 2) * 2)
        signals.append("many_inline_code_spans")

    # Enumerations / step-by-step markers
    enum_hits = len(re.findall(r"(?m)^\s*(\d+[\.\)]|\([0-9]+\))\s+", desc))
    if enum_hits:
        points += min(30, enum_hits * 6)
        signals.append("enumerated_steps")

    cond_hits = len(re.findall(r"\b(if|when|unless|otherwise)\b", desc, flags=re.I))
    if cond_hits:
        points += min(30, cond_hits * 3)
        signals.append("many_conditionals")

    # "algorithm words" that often make the fix straightforward for LLMs
    alg_words = [
        "split","normalize","cache","cached","thread","lock","mutex","atomic",
        "drive","importable","importlib","regex","pattern","validate","validation",
        "reject","raise","error","invariant","deterministic","keyed","key",
    ]
    aw = 0
    dl = desc.lower()
    for w in alg_words:
        if w in dl:
            aw += 1
    if aw:
        points += min(35, aw * 3)
        signals.append("algorithmic_keywords")

    # Spec-mirror overlap (tests echo the description)
    if overlap >= 0.55:
        points += 35
        signals.append(f"spec_mirror_overlap_{overlap:.2f}")
    elif overlap >= 0.40:
        points += 18
        signals.append(f"partial_spec_overlap_{overlap:.2f}")

    # Convert points -> penalty on hardness score (0..100). Higher penalty => easier => higher predicted pass.
    # We clamp aggressively because your current heuristic is underestimating easy problems.
    penalty = min(80.0, points * 0.85)

    return {
        "points": int(points),
        "spec_overlap": float(overlap),
        "penalty": float(penalty),
        "signals": signals,
    }


def _estimate_ai_pass_rate_heuristic(*, description_text: Optional[str], test_patch_text: str, policy: Any = None) -> Dict[str, Any]:
    if policy is None:
        policy = PolicyOptions()

    new_test_path = _infer_new_test_path_from_patch(test_patch_text)

    test_text = ""
    if new_test_path:
        try:
            test_text = _extract_new_file_content_from_patch(test_patch_text, new_test_path) or ""
        except Exception:
            test_text = ""
    if not test_text:
        test_text = test_patch_text

    dlint = _description_lint(description_text or "") if description_text is not None else None
    metrics = _simple_test_metrics(test_text)

    # hardness score: 0..100 (higher = harder -> lower predicted AI pass)
    # Sub-linear scaling: many tests/asserts increase difficulty, but not linearly.
    score = 40.0
    score += 6.0 * math.log1p(metrics["num_tests"])
    score += 2.0 * math.log1p(metrics["asserts"])
    if metrics["raises_ctx"]:
        score += 6.0
    if metrics["parametrize"]:
        score += 5.0
    if metrics["hypothesis"]:
        score += 10.0
    if metrics["bool_traps"]:
        score += 7.0
    if metrics["monkeypatch"]:
        score += 3.0

    # penalize hinty / overly internal descriptions (makes it easier for AI and less fair)
    if dlint:
        score -= min(12.0, 4.0 * len(dlint.get("solution_hint_patterns") or []))
        score -= min(10.0, max(0, len(dlint.get("file_mentions") or []) - 3) * 1.0)
        score -= min(8.0, max(0, len(dlint.get("dotted_identifiers") or []) - 6) * 0.5)

    score = max(0.0, min(100.0, score))

# map score -> pass probability (simple linear mapping)
    p = 0.90 - (score / 100.0)
    p = max(0.01, min(0.99, p))

    hintiness = _hintiness_diagnose(description_text, test_patch_text)
    adjusted_score = max(0.0, score - float(hintiness.get('penalty', 0.0)))
    p_ai = _ai_score_to_pass_rate(adjusted_score, policy)
    return {
        "hardness_score_0_100": score,
        "hintiness": hintiness,
        "predicted_ai_pass_rate_0_1": p_ai,
        "simulation": _ai_simulate_trials(
            pass_rate_0_1=p_ai,
            key=(description_text or "") + "\n" + (test_patch_text or ""),
            trials=int(getattr(policy, "ai_sim_trials", 5)),
        ),
        "new_test_path": new_test_path,
        "description_lint": dlint,
        "test_metrics": metrics,
        "notes": [
            "Heuristic only (no network / no real LLM runs).",
            "Tune policy.ai_target_pass_rate_* to match your workflow.",
        ],
    }

def _static_patch_checks(
    *,
    patch_text: str,
    kind: Literal["test.patch", "solution.patch"],
    policy: PolicyOptions,
) -> Tuple[List[Check], Dict[str, Any]]:
    """
    Returns (checks, meta).
    meta includes: changed_files (list[str]), file_map (dict).
    """
    checks: List[Check] = []
    meta: Dict[str, Any] = {"changed_files": [], "issues": []}

    # Patch must be valid UTF-8 and LF-only; ASCII optionally.
    ascii_issues = _scan_ascii_lf(kind, patch_text, enforce=policy.enforce_ascii_lf)
    checks.append(
        _check(
            f"{kind} encoding",
            passed=(len(ascii_issues) == 0),
            description="Patch text must be UTF-8, ASCII-only (workflow), and LF-only.",
            message="OK" if not ascii_issues else "; ".join(ascii_issues),
            details={"issues": ascii_issues},
        )
    )

    if not patch_text.strip():
        checks.append(
            _check(
                f"{kind} non-empty",
                passed=False,
                description="Patch must not be empty.",
                message="Patch is empty",
                details={},
            )
        )
        return checks, meta

    if "diff --git " not in patch_text:
        checks.append(
            _check(
                f"{kind} unified diff",
                passed=False,
                description="Patch must be a unified diff (expected 'diff --git' headers).",
                message="Missing 'diff --git' headers",
                details={},
            )
        )
        return checks, meta

    file_map = _parse_changed_files_unified_diff(patch_text)
    changed = sorted(file_map.keys())
    meta["changed_files"] = changed
    meta["file_map"] = file_map

    if kind == "test.patch":
        allowed: List[str] = ["test.sh"]
        allowed_prefixes: List[str] = ["tests/"]

        def _is_allowed(path: str) -> bool:
            if path in allowed:
                return True
            if any(path.startswith(p) for p in allowed_prefixes):
                return True
            return False

        forbidden = [p for p in changed if not _is_allowed(p)]
        checks.append(
            _check(
                "test.patch only touches allowed paths",
                passed=(len(forbidden) == 0),
                description="Enforce tests-only boundary: only test.sh and tests/ are allowed in test.patch.",
                message="OK" if not forbidden else "Forbidden paths: " + ", ".join(forbidden),
                details={"forbidden_paths": forbidden, "changed_files": changed},
            )
        )

        # Single new test file rule (strict default).
        test_py = [p for p in changed if p.startswith("tests/") and p.endswith(".py")]
        new_added = [p for p in test_py if file_map.get(p, {}).get("is_new")]
        # If diff parser missed is_new, fall back to /dev/null marker.
        if not new_added:
            for p in test_py:
                if f"diff --git a/{p} b/{p}" in patch_text and "--- /dev/null" in patch_text.split(f"diff --git a/{p} b/{p}", 1)[1].split("diff --git", 1)[0]:
                    new_added.append(p)

        # Enforce exactly one new test file and (optionally) naming.
        new_py = [p for p in new_added if p.startswith("tests/") and p.endswith(".py")]
        extra_tests = [p for p in test_py if p not in new_py]
        ok_count = (len(new_py) == 1)
        checks.append(
            _check(
                "test.patch adds exactly one new tests/*.py",
                passed=((not policy.enforce_single_new_test_file) or ok_count),
                description="Workflow rule: add exactly one new test file directly under tests/.",
                message="OK" if ok_count else f"Expected 1 new tests/*.py, found {len(new_py)}",
                details={"new_test_files": new_py, "other_tests_py": extra_tests},
            )
        )

        if policy.enforce_new_test_suffix_problem and len(new_py) == 1:
            fn = Path(new_py[0]).name
            ok_name = (fn.startswith("test_") and fn.endswith("_problem.py"))
            checks.append(
                _check(
                    "new test file name ends with _problem.py",
                    passed=ok_name,
                    description="Workflow naming rule: tests/test_*_problem.py",
                    message="OK" if ok_name else f"Bad file name: {fn}",
                    details={"new_test_file": new_py[0]},
                )
            )

        # test.sh semantics (best-effort).
        if "test.sh" in changed:
            content = _extract_new_file_content_from_patch(patch_text, "test.sh")
            if content is None:
                # Might be a modification; do a heuristic scan over patch chunk.
                # We at least enforce no tail piping in the patch itself.
                content = patch_text
                checks.append(
                    _check_warn(
                        "test.sh content extractable",
                        passed=False,
                        description="Could not confidently reconstruct test.sh content from patch; heuristics will be weaker.",
                        message="Heuristic scan only",
                        details={},
                    )
                )

            # Accept both "case $1 in" and "if/elif on $1" styles.
            has_selector = ("$1" in content) or ("${1" in content) or ("\"$1\"" in content)
            has_case = bool(re.search(r"\bcase\s+.*\b\$1\b", content))
            has_base = bool(
                re.search(r"\bbase\)", content)  # case label
                or re.search(r'([=]{1,2}|!=)\s*["\']base["\']', content)  # string compare
                or re.search(r'\bbase\b.*\bthen\b', content)  # loose if-then
            )
            has_new = bool(
                re.search(r"\bnew\)", content)
                or re.search(r'([=]{1,2}|!=)\s*["\']new["\']', content)
                or re.search(r'\bnew\b.*\bthen\b', content)
            )
            has_split = bool(has_selector and has_base and has_new)
            checks.append(
                _check(
                    "test.sh has base/new dispatch",
                    passed=has_split,
                    description="test.sh must support ./test.sh base and ./test.sh new (case or if/elif dispatch).",
                    message="OK" if has_split else "Missing base/new dispatch logic",
                    details={"has_selector": has_selector, "has_case": has_case, "has_base": has_base, "has_new": has_new},
                )
            )

            if policy.forbid_tail_piping_in_test_sh:
                has_tail = bool(re.search(r"\|\s*tail\b", content))
                checks.append(
                    _check(
                        "test.sh must not pipe to tail",
                        passed=(not has_tail),
                        description="Avoid runner output manipulation (validator/reviewer often flags this).",
                        message="OK" if not has_tail else "Found '| tail' piping",
                        details={"matched": has_tail},
                    )
                )

            # If we can determine the new test file path, enforce ignore/run semantics.            new_test_path = new_py[0] if len(new_py) == 1 else None
            new_test_path = None
            if new_test_path and content is not None:
                # If we couldn't reconstruct test.sh content (common for modifications),
                # fall back to scanning the test.sh diff block. If we still can't find a
                # definitive signal, downgrade to INFO rather than producing noisy WARNs.
                diff_block = _extract_file_diff_block(patch_text, "test.sh") or ""
                haystack = content
                heuristic_only = (haystack is patch_text)  # set earlier when extract failed
                if heuristic_only and diff_block:
                    haystack = diff_block

                base_ignores = bool(re.search(rf"--ignore(=|\s+){re.escape(new_test_path)}\b", haystack))
                new_runs_only = bool(re.search(rf"\bpytest\b.*\b{re.escape(new_test_path)}\b", haystack))

                sev_if_missing = "INFO" if heuristic_only else "WARN"
                checks.append(
                    _check(
                        "test.sh base ignores the new test file",
                        passed=base_ignores,
                        description="base mode must exclude the new test file via --ignore=tests/....py",
                        message="OK" if base_ignores else ("Not confidently detected from patch (heuristic-only)" if heuristic_only else "Did not find --ignore=<new test file> in base branch"),
                        details={"new_test_file": new_test_path, "heuristic_only": heuristic_only},
                        severity_if_fail=sev_if_missing,
                    )
                )
                checks.append(
                    _check(
                        "test.sh new runs only the new test file",
                        passed=new_runs_only,
                        description="new mode should target only the new test file.",
                        message="OK" if new_runs_only else ("Not confidently detected from patch (heuristic-only)" if heuristic_only else "Did not find pytest invocation targeting the new test file"),
                        details={"new_test_file": new_test_path, "heuristic_only": heuristic_only},
                        severity_if_fail=sev_if_missing,
                    )
                )
    else:
        # solution.patch path constraints
        forbidden_paths: List[str] = []
        if policy.forbid_solution_touch_tests:
            forbidden_paths.extend([p for p in changed if p == "test.sh" or p.startswith("tests/")])
        if policy.forbid_solution_touch_dockerfiles:
            forbidden_paths.extend([p for p in changed if Path(p).name.lower().startswith("dockerfile")])

        # Also forbid patch touching patch files and metadata/config that isn't source.
        forbidden_paths.extend([p for p in changed if p in {"test.patch", "solution.patch"}])
        forbidden_paths.extend([p for p in changed if p.startswith(".github/") or p in {"pyproject.toml", "setup.cfg", "tox.ini"}])

        forbidden_paths = sorted(set(forbidden_paths))
        checks.append(
            _check(
                "solution.patch touches only production code",
                passed=(len(forbidden_paths) == 0),
                description="solution.patch must not modify tests/test.sh/Dockerfile/config/patch files.",
                message="OK" if not forbidden_paths else "Forbidden paths: " + ", ".join(forbidden_paths),
                details={"forbidden_paths": forbidden_paths, "changed_files": changed},
            )
        )
        # Lightweight suspicious-pattern scan (WARN): this is not a hard fail, but helps catch
        # obvious "run commands / download things" smells that reviewers often notice.
        suspicious_hits: List[str] = []
        suspicious_patterns = [
            r"\bcurl\b",
            r"\bwget\b",
            r"\bapt-get\b",
            r"\byum\b",
            r"\bapk\b",
            r"\bpacman\b",
            r"\bbrew\b",
            r"\bgit\s+clone\b",
            r"\bssh\b",
            r"\bscp\b",
            r"\bnetcat\b|\bnc\b",
        ]
        for pat in suspicious_patterns:
            if re.search(pat, patch_text, flags=re.IGNORECASE):
                suspicious_hits.append(pat)
        checks.append(
            _check_warn(
                "solution.patch suspicious pattern scan",
                passed=(len(suspicious_hits) == 0),
                description="Warns on common shell/network keywords in the patch text (reviewer-preflight).",
                message="OK" if not suspicious_hits else "Suspicious keywords found: " + ", ".join(suspicious_hits),
                details={"hits": suspicious_hits},
            )
        )


    return checks, meta


def run_static_postchecks(
    *,
    dockerfile_text: str,
    test_patch_text: str,
    solution_patch_text: Optional[str],
    description_text: Optional[str],
    policy: PolicyOptions,
) -> Dict[str, Any]:
    """
    Backwards-compatible static response (with extra fields).
    """
    all_checks: List[Check] = []
    docker_checks = _dockerfile_checks(dockerfile_text)
    all_checks.extend(docker_checks)

    test_checks, test_meta = _static_patch_checks(patch_text=test_patch_text, kind="test.patch", policy=policy)
    all_checks.extend(test_checks)

    sol_checks: List[Check] = []
    sol_meta: Dict[str, Any] = {"changed_files": [], "issues": []}
    if solution_patch_text is not None:
        sol_checks, sol_meta = _static_patch_checks(patch_text=solution_patch_text, kind="solution.patch", policy=policy)
        all_checks.extend(sol_checks)
    else:
        sim = (ai_result.get('simulation') or {}) if isinstance(ai_result, dict) else {}
        sim_passes = sim.get('passes')
        sim_trials = sim.get('trials')
        sim_passes_max = int(getattr(policy, 'ai_sim_passes_max', 1))
        sim_ok = (sim_passes is None) or (int(sim_passes) <= sim_passes_max)
        sim_suffix = (
            f"; sim_passes={sim_passes}/{sim_trials} (max {sim_passes_max})"
            if (sim_passes is not None and sim_trials is not None) else ""
        )
        all_checks.append(
            _check(
                'AI Difficulty Estimate',
                passed=bool(in_band and sim_ok),
                description='Heuristic estimate of AI pass rate based on description+tests structure (no real LLM runs).',
                message=(
                    (f'predicted_ai_pass_rate~{p_ai} (target {band_min:.2f}..{band_max:.2f})' + sim_suffix)
                    if p_ai is not None else 'Could not compute estimate'
                ),
                details={'target_band': [band_min, band_max], **ai_result, 'ai': dict(ai_result), 'sim_passes_max': sim_passes_max},
                severity_if_fail=sev,
            )
        )

    desc_issues: List[str] = []
    if description_text is not None:
        desc_issues.extend(_scan_ascii_lf("description", description_text, enforce=policy.enforce_ascii_lf))
        if policy.enforce_no_test_leakage_in_description:
            for pat in _LEAKY_DESC_PATTERNS:
                if re.search(pat, description_text, flags=re.IGNORECASE):
                    desc_issues.append(f"description: contains test leakage marker matching /{pat}/")
                    break

    all_checks.append(
        _check_warn(
            "Problem description hygiene",
            passed=(len(desc_issues) == 0),
            description="ASCII/LF checks (and optional test-leakage scan) for the description text.",
            message="OK" if not desc_issues else "; ".join(desc_issues),
            details={"issues": desc_issues},
        )
    )

    # Solution patch format check: ensure no overlap with test.patch changed files.
    overlap = sorted(set(test_meta.get('changed_files', [])) & set(sol_meta.get('changed_files', [])))
    all_checks.append(
        _check(
            'Solution Patch Format Check',
            passed=(len(overlap) == 0),
            description='solution.patch must not overlap any files touched by test.patch.',
            message='OK' if not overlap else ('overlap: ' + ', '.join(overlap)),
            details={'overlap_files': overlap},
            severity_if_fail='FAIL',
        )
    )

    # Description presence / hint hygiene (configurable).
    if bool(getattr(policy, 'enforce_description_present', False)):
        sim = (ai_result.get('simulation') or {}) if isinstance(ai_result, dict) else {}
        sim_passes = sim.get('passes')
        sim_trials = sim.get('trials')
        sim_passes_max = int(getattr(policy, 'ai_sim_passes_max', 1))
        sim_ok = (sim_passes is None) or (int(sim_passes) <= sim_passes_max)
        sim_suffix = (
            f"; sim_passes={sim_passes}/{sim_trials} (max {sim_passes_max})"
            if (sim_passes is not None and sim_trials is not None) else ""
        )
        all_checks.append(
            _check(
                'AI Difficulty Estimate',
                passed=bool(in_band and sim_ok),
                description='Heuristic estimate of AI pass rate based on description+tests structure (no real LLM runs).',
                message=(
                    (f'predicted_ai_pass_rate~{p_ai} (target {band_min:.2f}..{band_max:.2f})' + sim_suffix)
                    if p_ai is not None else 'Could not compute estimate'
                ),
                details={'target_band': [band_min, band_max], **ai_result, 'ai': dict(ai_result), 'sim_passes_max': sim_passes_max},
                severity_if_fail=sev,
            )
        )

    ai_result: Dict[str, Any] = {}
    if bool(getattr(policy, 'ai_heuristics_enabled', True)):
        ai_result = _estimate_ai_pass_rate_heuristic(description_text=description_text, test_patch_text=test_patch_text, policy=policy)
        # Gate: deterministic simulation (default: 5 trials, allow <= 1 pass)
        sim_trials = int(getattr(policy, 'ai_sim_trials', 5))
        sim_passes_max = int(getattr(policy, 'ai_sim_passes_max', 1))
        sim = (ai_result or {}).get('simulation') if isinstance(ai_result, dict) else None
        if not isinstance(sim, dict):
            try:
                pr = float((ai_result or {}).get('predicted_ai_pass_rate_0_1') or 0.5)
            except Exception:
                pr = 0.5
            sim = _ai_simulate_trials(pass_rate_0_1=pr, key=(description_text or '') + '\n' + (test_patch_text or ''), trials=sim_trials)
            if isinstance(ai_result, dict):
                ai_result['simulation'] = sim
        passes = sim.get('passes') if isinstance(sim, dict) else None
        trials = sim.get('trials') if isinstance(sim, dict) else sim_trials
        ok_sim = isinstance(passes, int) and passes <= sim_passes_max
        sev_sim = 'FAIL' if bool(getattr(policy, 'ai_enforce_simulation', True)) else 'WARN'
        all_checks.append(
            _check(
                'AI Trial Simulation',
                passed=bool(ok_sim),
                description='Deterministic N-trial simulation (no real LLM). Default: <=1 pass out of 5.',
                message=(('passes=%s/%s (max %s)' % (passes, trials, sim_passes_max)) if passes is not None else 'Could not simulate'),
                details={'simulation': sim, 'sim_trials': sim_trials, 'sim_passes_max': sim_passes_max},
                severity_if_fail=sev_sim,
            )
        )
        # Optional: description must contain enough explicit requirements (must/must not)
        try:
            dlint = (ai_result or {}).get('description_lint') or {}
            req_words = int(dlint.get('requirements_word_count') or 0)
        except Exception:
            dlint = {}
            req_words = 0
        min_req = int(getattr(policy, 'description_min_requirements', 6))
        ok_req = req_words >= min_req
        sev_req = 'FAIL' if bool(getattr(policy, 'enforce_description_min_requirements', False)) else 'WARN'
        all_checks.append(
            _check(
                'Description requirement density',
                passed=bool(ok_req),
                description='Heuristic alignment: description should include enough explicit requirements (must/must not).',
                message=('requirements_word_count=%s (min %s)' % (req_words, min_req)),
                details={'requirements_word_count': req_words, 'min_required': min_req, **(dlint if isinstance(dlint, dict) else {})},
                severity_if_fail=sev_req,
            )
        )
        ai_result_nested = json.loads(json.dumps(ai_result))
        band_min = float(getattr(policy, 'ai_target_pass_rate_min', 0.10))
        band_max = float(getattr(policy, 'ai_target_pass_rate_max', 0.50))
        p_ai = ai_result.get('predicted_ai_pass_rate_0_1')
        in_band = isinstance(p_ai, (int, float)) and band_min <= float(p_ai) <= band_max
        sev = 'FAIL' if bool(getattr(policy, 'ai_enforce_target_band', False)) else 'WARN'
        all_checks.append(
            _check(
                'AI Difficulty Estimate',
                passed=bool(in_band),
                description='Heuristic estimate of AI pass rate based on description+tests structure (no real LLM runs).',
                message=(f'predicted_ai_pass_rate~{p_ai} (target {band_min:.2f}..{band_max:.2f})' if p_ai is not None else 'Could not compute estimate'),
                details={'target_band': [band_min, band_max], **ai_result, 'ai': ai_result_nested},
                severity_if_fail=sev,
            )
        )

        if bool(getattr(policy, 'enforce_no_solution_hints_in_description', True)) and description_text is not None:
            dlint = ai_result.get('description_lint') or {}
            hints = dlint.get('solution_hint_patterns') or []
            all_checks.append(
                _check(
                    'Problem description avoids solution hints',
                    passed=(len(hints) == 0),
                    description='Description should document requirements and constraints, not how to implement the fix.',
                    message='OK' if not hints else ('Found hint patterns: ' + ', '.join(hints)),
                    details=dlint,
                    severity_if_fail='WARN',
                )
            )

    # Backwards-compatible aggregation.
    def _compact(ck: Check) -> Dict[str, Any]:
        report = build_validator_style_report_from_checks(all_checks, include_execution_placeholders=True)
        # description coverage (fairness / undocumented requirements)
        desc_cov = _compute_description_coverage(description_text=description_text, test_patch_text=test_patch_text, policy=policy)
        missing = desc_cov.get("missing_concepts") or []
        cov_passed = len(missing) == 0
        enforce = bool(getattr(policy, 'enforce_description_coverage', False))
        sev = 'FAIL' if (enforce and not cov_passed) else ('WARN' if not cov_passed else 'INFO')
        msg = 'OK' if cov_passed else ('Missing concept(s) in description: ' + ', '.join(missing))
        report['Problem description coverage'] = {
            'Description': 'Heuristic fairness guard: flags concepts exercised by tests that appear undocumented in the description.',
            'Message': msg,
            'Details': desc_cov,
            'Severity': sev,
        }
        return {
        "effective_policy": {
            "ai_curve_center": getattr(policy, "ai_curve_center", None),
            "ai_curve_slope": getattr(policy, "ai_curve_slope", None),
            "ai_target_pass_rate_min": getattr(policy, "ai_target_pass_rate_min", None),
            "ai_target_pass_rate_max": getattr(policy, "ai_target_pass_rate_max", None),
        },

    "ai": ai_result,
            "name": ck.name,
            "passed": ck.passed,
            "severity": ck.severity,
            "message": ck.message,
            "details": ck.details,
        }

    # Old field "all_issues" as a string.
    issues_flat: List[str] = []
    for ck in all_checks:
        if not ck.passed and ck.severity in {"FAIL", "WARN"}:
            issues_flat.append(f"[{ck.severity}] {ck.name}: {ck.message}")

    ok = all(ck.passed or ck.severity in {"WARN", "INFO"} for ck in all_checks)

    all_checks = list(locals().get("all_checks", [])) or (list(docker_checks) + list(test_checks) + list(sol_checks))
    report = build_validator_style_report_from_checks(all_checks, include_execution_placeholders=True)
    return {
        "ok": ok,
        "dockerfile": {
            "checks": [_compact(c) for c in docker_checks],
        },
        "test_patch": {
            "changed_files": test_meta.get("changed_files", []),
            "checks": [_compact(c) for c in test_checks],
        },
        "solution_patch": {
            "changed_files": sol_meta.get("changed_files", []),
            "checks": [_compact(c) for c in sol_checks],
        },
        "description": {
            "present": description_text is not None,
            "issues": desc_issues,
        },
        "all_issues": "\n".join(issues_flat),
        "report": report,

    }


def build_validator_style_report_from_checks(checks: List[Check], *, include_execution_placeholders: bool) -> Dict[str, Any]:
    """
    Produce "validator-style report" (sectioned dict of checks).
    Shape:
      { "<Check Name>": { "Description": ..., "Message": ..., "Details": {...}, "Severity": ... }, ... }
    """
    report: Dict[str, Any] = {}
    # Keep stable ordering:
    for ck in checks:
        report[ck.name] = {
            "Description": ck.description,
            "Message": ck.message,
            "Details": ck.details,
            "Severity": ck.severity,
        }

    if include_execution_placeholders:
        report.setdefault(
            "Problem Test Execution",
            {
                "Description": "Runs ./test.sh base (must pass) and ./test.sh new (must fail) on base code with test.patch applied.",
                "Message": "Not run (static-only endpoint).",
                "Details": {"status": "not_run"},
                "Severity": "INFO",
            },
        )
        report.setdefault(
            "Solution Test Execution",
            {
                "Description": "Runs ./test.sh base and ./test.sh new after applying solution.patch (both must pass).",
                "Message": "Not run (static-only endpoint).",
                "Details": {"status": "not_run"},
                "Severity": "INFO",
            },
        )

    return report


# -----------------------------
# Triad runner
# -----------------------------

@dataclass
class CmdResult:
    rc: int
    duration_sec: float
    last_output: str
    cmd: List[str]
    cwd: str


_PYTEST_SUMMARY_RE = re.compile(
    r"(?:(?P<banner>=+)\s*)?"
    r"(?:(?P<passed>\d+)\s+passed)?"
    r"(?:,?\s*(?P<failed>\d+)\s+failed)?"
    r"(?:,?\s*(?P<errors>\d+)\s+errors)?"
    r"(?:,?\s*(?P<skipped>\d+)\s+skipped)?"
    r"(?:,?\s*(?P<xfailed>\d+)\s+xfailed)?"
    r"(?:,?\s*(?P<xpassed>\d+)\s+xpassed)?"
    r".*?\bin\s+(?P<secs>\d+(?:\.\d+)?)s\b",
    re.IGNORECASE,
)

def _parse_pytest_summary(text: str) -> Dict[str, Any]:
    if not text:
        return {"found": False}

    window = text[-200_000:]
    best_line = None

    # Find the last line that looks like a pytest summary.
    for line in reversed(window.splitlines()):
        line_s = line.strip()
        if not line_s:
            continue
        if " in " not in line_s:
            continue
        if not re.search(r"\bin\s+\d+(?:\.\d+)?s\b", line_s, flags=re.IGNORECASE):
            continue
        if not re.search(r"\b(passed|failed|errors?|skipped|xfailed|xpassed)\b", line_s, flags=re.IGNORECASE):
            continue
        best_line = line_s
        break

    if not best_line:
        return {"found": False}

    counts = {"passed": 0, "failed": 0, "errors": 0, "skipped": 0, "xfailed": 0, "xpassed": 0}
    for n, key in re.findall(r"(\d+)\s+(passed|failed|errors?|skipped|xfailed|xpassed)\b", best_line, flags=re.IGNORECASE):
        k = key.lower()
        if k == "error":
            k = "errors"
        counts[k] += int(n)

    m = re.search(r"\bin\s+(\d+(?:\.\d+)?)s\b", best_line, flags=re.IGNORECASE)
    dur = float(m.group(1)) if m else None

    out = {"found": True, "summary_line": best_line}
    out.update(counts)
    if dur is not None:
        out["reported_duration_sec"] = dur
    return out


def _parse_pytest_summary(text: str) -> Dict[str, Any]:
    if not text:
        return {"found": False}

    window = text[-120_000:]  # keep it fast even for large logs
    m_last = None
    for m in _PYTEST_SUMMARY_RE.finditer(window):
        m_last = m

    if not m_last:
        # fallback: scan line-by-line from the end
        for line in reversed(window.splitlines()):
            m = _PYTEST_SUMMARY_RE.search(line)
            if m:
                m_last = m
                break

    if not m_last:
        return {"found": False}

    def _i(name: str) -> int:
        v = m_last.groupdict().get(name)
        return int(v) if v else 0

    # Capture the full line that contained the match (best-effort).
    start = window.rfind("\n", 0, m_last.start())
    start = 0 if start < 0 else start + 1
    end = window.find("\n", m_last.end())
    end = len(window) if end < 0 else end
    summary_line = window[start:end].strip()

    return {
        "found": True,
        "passed": _i("passed"),
        "failed": _i("failed"),
        "errors": _i("errors"),
        "skipped": _i("skipped"),
        "xfailed": _i("xfailed"),
        "xpassed": _i("xpassed"),
        "reported_duration_sec": float(m_last.group("secs")),
        "summary_line": summary_line,
    }

def _run_cmd(
    *,
    job_dir: Path,
    cmd: List[str],
    cwd: Path,
    env: Optional[Dict[str, str]] = None,
    log_names: Iterable[str] = ("triad.log", "runner.log"),
    capture_lines: int = 2000,
) -> CmdResult:
    start = time.perf_counter()
    _log(job_dir, "runner.log", f"$ (cwd={cwd}) " + " ".join(cmd))

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )
    except FileNotFoundError as e:
        # Common when docker/git are missing from PATH.
        msg = f"{type(e).__name__}: {e}"
        for ln in log_names:
            _append_log(job_dir / ln, msg)
        dur = time.perf_counter() - start
        return CmdResult(rc=127, duration_sec=dur, last_output=msg + "\n", cmd=cmd, cwd=str(cwd))

    buf: Deque[str] = deque(maxlen=capture_lines)
    assert proc.stdout is not None
    for line in proc.stdout:
        buf.append(line)
        for ln in log_names:
            _append_log(job_dir / ln, line.rstrip("\n"))

    rc = proc.wait()
    dur = time.perf_counter() - start
    return CmdResult(rc=rc, duration_sec=dur, last_output="".join(buf), cmd=cmd, cwd=str(cwd))


def _docker_build(
    *,
    job_dir: Path,
    repo_dir: Path,
    dockerfile_text: str,
    image_tag: str,
) -> CmdResult:
    # Copy Dockerfile into context (best compatibility).
    df_path = repo_dir / "Dockerfile.problem"
    _write_text_file(df_path, dockerfile_text)
    # also keep a copy in job artifacts:
    _write_text_file(job_dir / "Dockerfile.problem", dockerfile_text)

    # Improve docker build performance by reducing context size.
    # Only create a .dockerignore when the repo does not provide one.
    di_path = repo_dir / ".dockerignore"
    if not di_path.exists():
        _write_text_file(
            di_path,
            """.git
.venv
venv
__pycache__
.pytest_cache
.mypy_cache
.ruff_cache
.tox
dist
build
*.pyc
""",
        )

    # Best-effort cache: if the image already exists, skip rebuilding.
    inspect = _run_cmd(
        job_dir=job_dir,
        cmd=["docker", "image", "inspect", image_tag],
        cwd=repo_dir,
        log_names=("docker_build.log", "triad.log", "runner.log"),
    )
    if inspect.rc == 0:
        return CmdResult(
            rc=0,
            duration_sec=inspect.duration_sec,
            last_output=(inspect.last_output + "\n[validator-local] docker image already present; skipping build\n"),
            cmd=inspect.cmd,
            cwd=inspect.cwd,
        )

    cmd = ["docker", "build", "-f", str(df_path), "-t", image_tag, str(repo_dir)]
    return _run_cmd(
        job_dir=job_dir,
        cmd=cmd,
        cwd=repo_dir,
        log_names=("docker_build.log", "triad.log", "runner.log"),
    )


def _docker_run_bash(
    *,
    job_dir: Path,
    image_tag: str,
    repo_dir: Path,
    bash_cmd: str,
) -> CmdResult:
    """Run a bash command inside the built image.

    Notes:
      - Forces offline networking ("--network none").
      - On Windows hosts, avoid "-u uid:gid" (Docker Desktop can reject it).
      - Normalize volume paths on Windows to forward slashes for docker CLI.
    """
    host_path = str(repo_dir)
    if os.name == "nt":
        host_path = host_path.replace("\\", "/")

    cmd = [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "-v",
        f"{host_path}:/app",
        "-w",
        "/app",
    ]

    if os.name != "nt" and hasattr(os, "getuid") and hasattr(os, "getgid"):
        cmd += ["-u", f"{os.getuid()}:{os.getgid()}"]

    cmd += [
        image_tag,
        "bash",
        "-lc",
        bash_cmd,
    ]
    return _run_cmd(job_dir=job_dir, cmd=cmd, cwd=repo_dir, log_names=("triad.log", "runner.log"))


def _host_run_bash(
    *,
    job_dir: Path,
    repo_dir: Path,
    bash_cmd: str,
) -> CmdResult:
    # Dockerless fallback: execute the bash command on the host Python environment.
    # This does NOT provide container isolation; it is intended for environments without docker.
    return _run_cmd(job_dir=job_dir, cmd=["bash", "-lc", bash_cmd], cwd=repo_dir, log_names=("triad.log", "runner.log"))



def _is_git_repo(path: Path) -> bool:
    # Fast check that works for both bare and non-bare repos.
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=str(path),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return res.returncode == 0 and res.stdout.strip() == "true"
    except FileNotFoundError:
        return False


def _git_has_commit(repo_dir: Path, sha: str) -> bool:
    try:
        res = subprocess.run(
            ["git", "cat-file", "-e", f"{sha}^{{commit}}"],
            cwd=str(repo_dir),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return res.returncode == 0
    except FileNotFoundError:
        return False


def _copy_tree_for_worktree(src: Path, dst: Path) -> None:
    def _ignore(dirpath: str, names: list[str]) -> set[str]:
        # Skip common virtualenv/build noise to speed up copies.
        drop = {
            ".git",
            ".venv",
            "venv",
            "__pycache__",
            ".pytest_cache",
            ".mypy_cache",
            ".ruff_cache",
            ".tox",
            "dist",
            "build",
        }
        return {n for n in names if n in drop}

    shutil.copytree(src, dst, symlinks=True, ignore=_ignore)


def _git_init_baseline(job_dir: Path, repo_dir: Path) -> str:
    # Initialize a lightweight git repo so we can apply patches with git-apply.
    _run_cmd(job_dir=job_dir, cmd=["git", "init"], cwd=repo_dir, log_names=("triad.log", "runner.log"))
    _run_cmd(job_dir=job_dir, cmd=["git", "add", "-A"], cwd=repo_dir, log_names=("triad.log", "runner.log"))
    res = _run_cmd(
        job_dir=job_dir,
        cmd=[
            "git",
            "-c",
            "user.name=validator-local",
            "-c",
            "user.email=validator-local@localhost",
            "commit",
            "-m",
            "baseline",
            "--no-gpg-sign",
        ],
        cwd=repo_dir,
        log_names=("triad.log", "runner.log"),
    )
    # It's ok if the repo has no changes (rare), but we still want a SHA.
    head = _run_cmd(job_dir=job_dir, cmd=["git", "rev-parse", "HEAD"], cwd=repo_dir, log_names=("triad.log", "runner.log"))
    if head.rc != 0:
        raise RuntimeError(f"git rev-parse HEAD failed after init (rc={head.rc})")
    return head.last_output.strip().splitlines()[-1].strip()


def _git_clone_checkout(
    *,
    job_dir: Path,
    repo_path: Path,
    sha: str,
    dest_dir: Path,
) -> str:
    """Prepare the repo worktree under dest_dir and return the resolved HEAD sha.

    Supports two source formats:
      - A normal git worktree (repo_path is a git repo): clones and optionally checks out `sha`.
      - A plain directory (no .git): copies files, initializes a git repo, and commits a baseline.
    """
    _ensure_dir(dest_dir.parent)

    sha = (sha or "HEAD").strip()

    if _is_git_repo(repo_path):
        # Validate sha early for better errors (common user mistake: providing a local synthetic SHA).
        if sha != "HEAD" and not _git_has_commit(repo_path, sha):
            raise RuntimeError(
                "requested sha is not present in the source repo. "
                "If you created a local git commit from a zip without history, that SHA will not exist elsewhere; "
                "use a real upstream commit SHA or use sha=HEAD for a plain folder input."
            )

        clone_log = job_dir / "git_clone.log"
        checkout_log = job_dir / "git_checkout.log"
        _log(job_dir, "runner.log", f"Cloning repo: {repo_path} -> {dest_dir}")

        # Local clones are faster and avoid copying objects when possible.
        clone_cmd = ["git", "clone", "--local", str(repo_path), str(dest_dir)]
        res1 = _run_cmd(
            job_dir=job_dir,
            cmd=clone_cmd,
            cwd=job_dir,
            log_names=(clone_log.name, "triad.log", "runner.log"),
        )
        if res1.rc != 0:
            # Fallback without --local (e.g., non-local URLs).
            res1b = _run_cmd(
                job_dir=job_dir,
                cmd=["git", "clone", str(repo_path), str(dest_dir)],
                cwd=job_dir,
                log_names=(clone_log.name, "triad.log", "runner.log"),
            )
            if res1b.rc != 0:
                raise RuntimeError(f"git clone failed (rc={res1b.rc})")

        if sha != "HEAD":
            _log(job_dir, "runner.log", f"Checking out SHA: {sha}")
            res2 = _run_cmd(
                job_dir=job_dir,
                cmd=["git", "checkout", sha],
                cwd=dest_dir,
                log_names=(checkout_log.name, "triad.log", "runner.log"),
            )
            if res2.rc != 0:
                raise RuntimeError(f"git checkout failed (rc={res2.rc})")

        head = _run_cmd(job_dir=job_dir, cmd=["git", "rev-parse", "HEAD"], cwd=dest_dir, log_names=("triad.log", "runner.log"))
        if head.rc != 0:
            raise RuntimeError(f"git rev-parse HEAD failed (rc={head.rc})")
        return head.last_output.strip().splitlines()[-1].strip()

    # Plain directory: copy and init.
    _log(job_dir, "runner.log", f"Preparing non-git repo directory: {repo_path} -> {dest_dir}")
    _copy_tree_for_worktree(repo_path, dest_dir)

    if sha != "HEAD":
        _log(job_dir, "runner.log", f"WARNING: repo_path is not a git repo; ignoring requested sha={sha!r} and using baseline HEAD")

    return _git_init_baseline(job_dir, dest_dir)


def _git_reset_clean(job_dir: Path, repo_dir: Path) -> None:
    _log(job_dir, "runner.log", "Resetting repo to clean state (git reset --hard; git clean -xdf)")
    _run_cmd(job_dir=job_dir, cmd=["git", "reset", "--hard"], cwd=repo_dir, log_names=("triad.log", "runner.log"))
    _run_cmd(job_dir=job_dir, cmd=["git", "clean", "-xdf"], cwd=repo_dir, log_names=("triad.log", "runner.log"))


def _git_apply_patch(job_dir: Path, repo_dir: Path, patch_text: str, artifact_name: str) -> None:
    patch_path = job_dir / artifact_name
    _write_text_file(patch_path, patch_text)
    _log(job_dir, "runner.log", f"Applying patch: {artifact_name}")
    res = _run_cmd(job_dir=job_dir, cmd=["git", "apply", "--check", str(patch_path)], cwd=repo_dir, log_names=("triad.log", "runner.log"))
    if res.rc != 0:
        raise RuntimeError(f"git apply --check failed for {artifact_name} (rc={res.rc})")
    res2 = _run_cmd(job_dir=job_dir, cmd=["git", "apply", str(patch_path)], cwd=repo_dir, log_names=("triad.log", "runner.log"))
    if res2.rc != 0:
        raise RuntimeError(f"git apply failed for {artifact_name} (rc={res2.rc})")

    # Ensure executable bit in worktree if possible.
    if artifact_name == "test.patch":
        p = repo_dir / "test.sh"
        if p.exists():
            try:
                p.chmod(0o755)
            except Exception:
                pass


def _update_status(job_dir: Path, updates: Dict[str, Any]) -> Dict[str, Any]:
    status_path = job_dir / "status.json"
    cur: Dict[str, Any] = {}
    if status_path.exists():
        cur = _json_load(status_path)
        if not isinstance(cur, dict):
            cur = {}
    cur.update(updates)
    cur.setdefault("job_id", job_dir.name)
    cur.setdefault("updated_at", _utc_now_iso())
    cur["updated_at"] = _utc_now_iso()
    _json_dump(status_path, cur)
    return cur


def _make_phase(result: CmdResult) -> Dict[str, Any]:
    return {
        "rc": result.rc,
        "duration_sec": result.duration_sec,
        "pytest": _parse_pytest_summary(result.last_output),
        "cmd": result.cmd,
        "cwd": result.cwd,
    }


def _triad(job_dir: Path, req: Dict[str, Any]) -> None:
    """
    Full triad runner. Writes artifacts and updates status.json continuously.
    Never raises to the API caller; exceptions become status=failed with crash.log.
    """
    _ensure_dir(job_dir)
    _update_status(job_dir, {"status": "running", "started_at": _utc_now_iso(), "reason": None, "crash_log": None})

    try:
        repo_path = Path(req["repo_path"]).expanduser().resolve()
        sha = str(req["sha"])
        dockerfile_text = str(req["dockerfile_text"])
        test_patch_text = str(req["test_patch_text"])
        solution_patch_text = str(req["solution_patch_text"])
        policy_dict = req.get("policy") or {}
        policy = PolicyOptions.model_validate(policy_dict)

        # Pre-triad static gate (fast fail).
        static = run_static_postchecks(
            dockerfile_text=dockerfile_text,
            test_patch_text=test_patch_text,
            solution_patch_text=solution_patch_text,
            description_text=req.get("description_text"),
            policy=policy,
        )
        _json_dump(job_dir / "static_postchecks.json", static)

        if not static.get("ok", False):
            _log(job_dir, "runner.log", "Static checks failed; skipping triad execution.")
            _update_status(job_dir, {"status": "failed", "finished_at": _utc_now_iso(), "reason": "static checks failed", "triad_summary": {"verdict": "FAIL", "headline": "static checks failed (see static_postchecks.json)"}})
            _json_dump(job_dir / "report.json", static.get("report", {}))
            return

        work_dir = job_dir / "work"
        repo_dir = work_dir / "repo"
        if work_dir.exists():
            shutil.rmtree(work_dir)
        _ensure_dir(work_dir)

        resolved_sha = _git_clone_checkout(job_dir=job_dir, repo_path=repo_path, sha=sha, dest_dir=repo_dir)

        df_hash = hashlib.sha256(dockerfile_text.encode("utf-8")).hexdigest()[:12]
        image_tag = f"validator-local:base-{resolved_sha[:12]}-{df_hash}"

        docker_on_path = shutil.which("docker") is not None
        execution_mode = "docker" if docker_on_path else "host"

        if docker_on_path:
            build_res = _docker_build(job_dir=job_dir, repo_dir=repo_dir, dockerfile_text=dockerfile_text, image_tag=image_tag)
            if build_res.rc != 0:
                raise RuntimeError(f"docker build failed (rc={build_res.rc})")
        else:
            # Still persist Dockerfile.problem into the repo context and job artifacts.
            df_path = repo_dir / "Dockerfile.problem"
            _write_text_file(df_path, dockerfile_text)
            _write_text_file(job_dir / "Dockerfile.problem", dockerfile_text)
            msg = "[validator-local] docker not found on PATH; skipping docker build and running tests on host python\n"
            _append_log(job_dir / "docker_build.log", msg.rstrip("\n"))
            _append_log(job_dir / "triad.log", msg.rstrip("\n"))
            _append_log(job_dir / "runner.log", msg.rstrip("\n"))
            build_res = CmdResult(rc=0, duration_sec=0.0, last_output=msg, cmd=["docker", "build", "<skipped>"], cwd=str(repo_dir))

        phases: Dict[str, Any] = {"docker_build": _make_phase(build_res), "execution_mode": execution_mode}

        def _run_phase_cmd(bash_cmd: str) -> CmdResult:
            if docker_on_path:
                return _docker_run_bash(job_dir=job_dir, image_tag=image_tag, repo_dir=repo_dir, bash_cmd=bash_cmd)
            return _host_run_bash(job_dir=job_dir, repo_dir=repo_dir, bash_cmd=bash_cmd)

        # Phase: apply test.patch, run base and new on base code.
        _git_apply_patch(job_dir, repo_dir, test_patch_text, "test.patch")

        _log(job_dir, "runner.log", "Phase: Problem Test Execution (base)")
        p_base = _run_phase_cmd("./test.sh base")
        phases["problem_base"] = _make_phase(p_base)

        _log(job_dir, "runner.log", "Phase: Problem Test Execution (new; expected FAIL)")
        p_new = _run_phase_cmd("./test.sh new")
        phases["problem_new"] = _make_phase(p_new)

        # Expected: base passes, new fails (nonzero).
        expected_fail_ok = (p_base.rc == 0) and (p_new.rc != 0)

        # Clean and apply test + solution for solution execution.
        _git_reset_clean(job_dir, repo_dir)
        _git_apply_patch(job_dir, repo_dir, test_patch_text, "test.patch")
        _git_apply_patch(job_dir, repo_dir, solution_patch_text, "solution.patch")

        _log(job_dir, "runner.log", "Phase: Solution Test Execution (base)")
        s_base = _run_phase_cmd("./test.sh base")
        phases["solution_base"] = _make_phase(s_base)

        _log(job_dir, "runner.log", "Phase: Solution Test Execution (new)")
        s_new = _run_phase_cmd("./test.sh new")
        phases["solution_new"] = _make_phase(s_new)

        ok_after_solution = (s_base.rc == 0 and s_new.rc == 0)

        # Verdict + headline.
        verdict = "OK" if (expected_fail_ok and ok_after_solution) else "FAIL"
        headline_parts: List[str] = []
        if expected_fail_ok:
            headline_parts.append("baseline passes; new tests fail before solution")
        else:
            if p_base.rc != 0:
                headline_parts.append("baseline FAILED (base should pass before solution)")
            if p_new.rc == 0:
                headline_parts.append("new unexpectedly PASSED on base (tests too weak or targeting wrong behavior)")
        if ok_after_solution:
            headline_parts.append("all pass after solution")
        else:
            if s_base.rc != 0:
                headline_parts.append("base FAILED after solution")
            if s_new.rc != 0:
                headline_parts.append("new FAILED after solution")

        headline = f"triad: {verdict} (" + "; ".join(headline_parts) + ")"

        triad_summary = {
            "verdict": verdict,
            "headline": headline,
            "execution_mode": phases.get("execution_mode"),
            "phases": phases,
        }
        _json_dump(job_dir / "triad_summary.json", triad_summary)

        # Build report (validator-style) from static + triad.
        report = build_validator_style_report_for_job(static, triad_summary)
        _json_dump(job_dir / "report.json", report)

        final_status = "succeeded" if verdict == "OK" else "failed"
        _update_status(
            job_dir,
            {
                "status": final_status,
                "finished_at": _utc_now_iso(),
                "reason": None if final_status == "succeeded" else headline,
                "triad_summary": triad_summary,
                "artifacts": sorted([p.name for p in job_dir.iterdir() if p.is_file()]),
            },
        )
        _log(job_dir, "triad.log", f"== triad: {verdict} ==")

    except Exception as e:
        # Always create crash.log and mark failed.
        _write_text_file(job_dir / "crash.log", f"{type(e).__name__}: {e}\n")
        _log(job_dir, "runner.log", f"Exception: {type(e).__name__}: {e}")
        _update_status(
            job_dir,
            {
                "status": "failed",
                "finished_at": _utc_now_iso(),
                "reason": f"{type(e).__name__}: {e}",
                "crash_log": "crash.log",
                "artifacts": sorted([p.name for p in job_dir.iterdir() if p.is_file()]),
            },
        )


def build_validator_style_report_for_job(static: Dict[str, Any], triad_summary: Dict[str, Any]) -> Dict[str, Any]:
    """
    Combine static checks + execution results into a validator-like report structure.
    """
    report: Dict[str, Any] = {}
    report["Static Checks Summary"] = {
        "Description": "Dockerfile + patch boundary/format checks.",
        "Message": "OK" if static.get("ok") else "Static checks failed",
        "Details": {"static_ok": static.get("ok"), "all_issues": static.get("all_issues", "")},
        "Severity": "PASS" if static.get("ok") else "FAIL",
    }

    # Carry over individual static checks in a stable order.
    static_report = static.get("report") or {}
    if isinstance(static_report, dict):
        for k, v in static_report.items():
            # Skip placeholders; we will add execution results below.
            if k in {"Problem Test Execution", "Solution Test Execution"}:
                continue
            report[k] = v

    phases = (triad_summary or {}).get("phases") or {}
    verdict = (triad_summary or {}).get("verdict", "FAIL")
    headline = (triad_summary or {}).get("headline", "")

    # Problem execution section
    pb = phases.get("problem_base") or {}
    pn = phases.get("problem_new") or {}
    pb_ok = (pb.get("rc") == 0)
    pn_expected_fail = (pn.get("rc") is not None and pn.get("rc") != 0)
    prob_ok = bool(pb_ok and pn_expected_fail)

    report["Problem Test Execution"] = {
        "Description": "Runs ./test.sh base (must pass) and ./test.sh new (must fail) on base code with test.patch applied.",
        "Message": "OK" if prob_ok else "Mismatch: base must pass and new must fail on base code",
        "Details": {
            "base": pb,
            "new": pn,
        },
        "Severity": "PASS" if prob_ok else "FAIL",
    }

    # Solution execution section
    sb = phases.get("solution_base") or {}
    sn = phases.get("solution_new") or {}
    sol_ok = (sb.get("rc") == 0 and sn.get("rc") == 0)
    report["Solution Test Execution"] = {
        "Description": "Runs ./test.sh base and ./test.sh new after applying solution.patch (both must pass).",
        "Message": "OK" if sol_ok else "FAIL: base/new must both pass after solution",
        "Details": {
            "base": sb,
            "new": sn,
        },
        "Severity": "PASS" if sol_ok else "FAIL",
    }

    report["Overall Triad Verdict"] = {
        "Description": "Final combined verdict for the triad runner.",
        "Message": headline or f"triad: {verdict}",
        "Details": {"verdict": verdict},
        "Severity": "PASS" if verdict == "OK" else "FAIL",
    }

    return report


# -----------------------------
# FastAPI app + in-memory job registry
# -----------------------------

app = FastAPI(title="validator-local", version="0.2.5")

_JOB_LOCK = threading.Lock()
_JOB_FUTURES: Dict[str, "concurrent.futures.Future[None]"] = {}
_EXECUTOR = None  # lazily created


def _get_executor():
    global _EXECUTOR
    if _EXECUTOR is None:
        import concurrent.futures
        _EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="validator_local")
    return _EXECUTOR


@app.on_event("startup")
def _startup() -> None:
    _ensure_dir(JOBS_DIR)


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "service": "validator-local",
        "version": app.version,
        "endpoints": [
            "GET /",
            "GET /healthz",
            "POST /v1/postchecks/static",
            "POST /v1/postchecks/static/from-files",
            "POST /v1/postchecks/report",
            "POST /v1/postchecks/report/from-files",
            "POST /v1/postchecks/ai",
            "POST /v1/postchecks/ai/from-files",
            "POST /v1/preflight/from-files",
            "POST /v1/jobs",
            "POST /v1/jobs/from-files",
            "GET /v1/jobs/{job_id}",
            "GET /v1/jobs/{job_id}/summary",
            "GET /v1/jobs/{job_id}/report",
            "GET /v1/jobs/{job_id}/artifacts",
            "GET /v1/jobs/{job_id}/artifacts/{name}",
            "GET /v1/jobs/{job_id}/artifacts/{name}/raw",
            "GET /v1/jobs/{job_id}/artifacts/{name}/tail?lines=N",
        ],
    }


@app.get("/healthz")
def healthz() -> Dict[str, Any]:
    return {"ok": True, "ts": _utc_now_iso()}


@app.post("/v1/postchecks/static")
def postchecks_static(req: StaticInlineRequest) -> Dict[str, Any]:
    out = run_static_postchecks(
        dockerfile_text=req.dockerfile_text,
        test_patch_text=req.test_patch_text,
        solution_patch_text=req.solution_patch_text,
        description_text=req.description_text,
        policy=req.policy,
    )
    if out.get("ai") is None:
        rep = out.get("report") or {}
        out["ai"] = (rep.get("AI Difficulty Estimate") or {}).get("Details") or {}
    return out


@app.post("/v1/postchecks/static/from-files")
def postchecks_static_from_files(req: StaticFromFilesRequest) -> Dict[str, Any]:
    dockerfile_text = _read_text_file(Path(req.dockerfile_path))
    test_patch_text = _read_text_file(Path(req.test_patch_path))
    sol_text = _read_text_file(Path(req.solution_patch_path)) if req.solution_patch_path else None
    desc_text = _read_text_file(Path(req.description_path)) if req.description_path else None

    out = run_static_postchecks(
        dockerfile_text=dockerfile_text,
        test_patch_text=test_patch_text,
        solution_patch_text=sol_text,
        description_text=desc_text,
        policy=req.policy,
    )
    if out.get("ai") is None:
        rep = out.get("report") or {}
        out["ai"] = (rep.get("AI Difficulty Estimate") or {}).get("Details") or {}
    return out


@app.post("/v1/preflight/from-files")
def preflight_from_files(req: JobFromFilesRequest) -> Dict[str, Any]:
    """Fast-fail endpoint: validates repo_path/sha/docker availability and runs static checks.

    This does NOT run docker build or tests. It is intended to catch doomed triad jobs early.
    """
    repo_path = Path(req.repo_path).expanduser().resolve()
    sha = str(req.sha)

    env: Dict[str, Any] = {
        "repo_path_exists": repo_path.exists(),
        "repo_path_is_dir": repo_path.is_dir(),
        "repo_is_git": _is_git_repo(repo_path),
        "docker_on_path": shutil.which("docker") is not None,
        "git_on_path": shutil.which("git") is not None,
        "sha": sha,
        "sha_resolvable": None,
        "notes": [],
    }

    # Extra environment diagnostics (fast, timeout-bounded).
    env["os"] = os.name
    env["resolved_sha"] = None
    env["docker_usable"] = None
    env["image_tag"] = None
    env["docker_image_present"] = None

    if env["repo_is_git"]:
        # Resolve the provided SHA (including HEAD) to a concrete commit for image-tag prediction.
        try:
            target = sha if sha else "HEAD"
            res = subprocess.run(
                ["git", "rev-parse", target],
                cwd=str(repo_path),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=2,
            )
            if res.returncode == 0:
                env["resolved_sha"] = res.stdout.strip().splitlines()[-1].strip()
        except Exception:
            pass

    if env["docker_on_path"]:
        try:
            res = subprocess.run(
                ["docker", "info"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=2,
            )
            env["docker_usable"] = (res.returncode == 0)
            if not env["docker_usable"]:
                env["notes"].append("docker is on PATH but the daemon is not reachable (docker info failed)")
        except Exception:
            env["docker_usable"] = False
            env["notes"].append("docker is on PATH but docker info timed out/errored; daemon may be unavailable")

    if env["repo_is_git"] and sha and sha != "HEAD":
        env["sha_resolvable"] = _git_has_commit(repo_path, sha)
        if not env["sha_resolvable"]:
            env["notes"].append(
                "sha is not present in repo_path. If you created a local commit from a zip without .git, that SHA won't exist elsewhere; use a real upstream SHA or set sha=HEAD."
            )
    elif not env["repo_is_git"]:
        env["sha_resolvable"] = (sha in ("", "HEAD"))
        if sha not in ("", "HEAD"):
            env["notes"].append("repo_path is not a git repo; sha will be ignored and a baseline commit will be created in the job workspace.")

    dockerfile_text = _read_text_file(Path(req.dockerfile_path))
    df_hash = hashlib.sha256(dockerfile_text.encode("utf-8")).hexdigest()[:12]
    if env.get("resolved_sha"):
        env["image_tag"] = f"validator-local:base-{str(env['resolved_sha'])[:12]}-{df_hash}"

    if env.get("docker_usable") and env.get("image_tag"):
        try:
            res = subprocess.run(
                ["docker", "image", "inspect", str(env["image_tag"])],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=2,
            )
            env["docker_image_present"] = (res.returncode == 0)
        except Exception:
            env["docker_image_present"] = None
    test_patch_text = _read_text_file(Path(req.test_patch_path))
    solution_patch_text = _read_text_file(Path(req.solution_patch_path))
    desc_text = _read_text_file(Path(req.description_path)) if getattr(req, "description_path", None) else None

    static = run_static_postchecks(
        dockerfile_text=dockerfile_text,
        test_patch_text=test_patch_text,
        solution_patch_text=solution_patch_text,
        description_text=desc_text,
        policy=req.policy,
    )

    ok_env = bool(env["repo_path_exists"] and env["repo_path_is_dir"] and env["git_on_path"])

    if not env["docker_on_path"]:
        env["notes"].append("docker is not on PATH; triad will run in host mode and will not execute Dockerfile.problem.")
    if env["repo_is_git"] and sha and sha != "HEAD":
        ok_env = ok_env and bool(env["sha_resolvable"])

    return {
        "ok": bool(static.get("ok", False)) and ok_env,
        "env": env,
        "static": static,
    }


@app.post("/v1/postchecks/report")
def postchecks_report(req: StaticInlineRequest) -> Dict[str, Any]:
    out = run_static_postchecks(
        dockerfile_text=req.dockerfile_text,
        test_patch_text=req.test_patch_text,
        solution_patch_text=req.solution_patch_text,
        description_text=req.description_text,
        policy=req.policy,
    )
    return out.get("report", {})


@app.post("/v1/postchecks/report/from-files")
def postchecks_report_from_files(req: StaticFromFilesRequest) -> Dict[str, Any]:
    out = postchecks_static_from_files(req)
    return out.get("report", {})


def _new_job_id() -> str:
    # timestamp + random suffix, sortable and readable.
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{uuid.uuid4().hex[:8]}"


def _job_dir(job_id: str) -> Path:
    return JOBS_DIR / job_id


def _write_request_artifact(job_dir: Path, obj: Dict[str, Any]) -> None:
    _json_dump(job_dir / "request.json", obj)


def _submit_job(job_id: str, req_obj: Dict[str, Any]) -> None:
    job_dir = _job_dir(job_id)
    _ensure_dir(job_dir)
    _write_request_artifact(job_dir, req_obj)
    _update_status(
        job_dir,
        {
            "status": "queued",
            "created_at": _utc_now_iso(),
            "reason": None,
            "crash_log": None,
            "triad_summary": None,
            "artifacts": sorted([p.name for p in job_dir.iterdir() if p.is_file()]),
        },
    )

    executor = _get_executor()
    fut = executor.submit(_triad, job_dir, req_obj)
    with _JOB_LOCK:
        _JOB_FUTURES[job_id] = fut


def _wait_for_job(job_id: str, timeout_sec: float) -> None:
    with _JOB_LOCK:
        fut = _JOB_FUTURES.get(job_id)
    if fut is None:
        # Job might have been created in a previous server process; fall back to polling status.json.
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            st = _json_load(_job_dir(job_id) / "status.json")
            if isinstance(st, dict) and st.get("status") in {"succeeded", "failed"}:
                return
            time.sleep(0.25)
        raise TimeoutError("timed out waiting for job completion")

    fut.result(timeout=timeout_sec)


def _read_status(job_id: str) -> Dict[str, Any]:
    job_dir = _job_dir(job_id)
    st = _json_load(job_dir / "status.json")
    if not isinstance(st, dict):
        return {"status": "failed", "reason": "corrupt status.json", "crash_log": "crash.log"}
    # Ensure never 500 by being defensive.
    st.setdefault("job_id", job_id)
    st.setdefault("status", "unknown")
    st.setdefault("artifacts", sorted([p.name for p in job_dir.iterdir() if p.is_file()]) if job_dir.exists() else [])
    return st


@app.post("/v1/jobs")
def create_job_inline(
    req: JobInlineRequest,
    wait: bool = Query(False, description="If true, block until job completes and return final status."),
    wait_timeout_sec: float = Query(DEFAULT_WAIT_TIMEOUT_SEC, ge=0.1, le=24 * 60 * 60),
) -> Dict[str, Any]:
    job_id = _new_job_id()
    job_dir = _job_dir(job_id)
    _ensure_dir(job_dir)

    req_obj = {
        "repo_path": req.repo_path,
        "sha": req.sha,
        "dockerfile_text": req.dockerfile_text,
        "test_patch_text": req.test_patch_text,
        "solution_patch_text": req.solution_patch_text,
        "description_text": None,
        "policy": req.policy.model_dump(),
    }

    # Persist inputs as artifacts (useful for debugging).
    _write_text_file(job_dir / "Dockerfile.problem", req.dockerfile_text)
    _write_text_file(job_dir / "test.patch", req.test_patch_text)
    _write_text_file(job_dir / "solution.patch", req.solution_patch_text)

    _submit_job(job_id, req_obj)

    effective_wait = wait or bool(req.wait)
    effective_timeout = req.wait_timeout_sec if req.wait_timeout_sec is not None else wait_timeout_sec

    if effective_wait:
        try:
            _wait_for_job(job_id, float(effective_timeout))
        except TimeoutError:
            st = _read_status(job_id)
            st["wait"] = {"completed": False, "timeout_sec": effective_timeout}
            return st

    return _read_status(job_id)


@app.post("/v1/jobs/from-files")
def create_job_from_files(
    req: JobFromFilesRequest,
    wait: bool = Query(False, description="If true, block until job completes and return final status."),
    wait_timeout_sec: float = Query(DEFAULT_WAIT_TIMEOUT_SEC, ge=0.1, le=24 * 60 * 60),
) -> Dict[str, Any]:
    job_id = _new_job_id()
    job_dir = _job_dir(job_id)
    _ensure_dir(job_dir)

    dockerfile_text = _read_text_file(Path(req.dockerfile_path))
    test_patch_text = _read_text_file(Path(req.test_patch_path))
    solution_patch_text = _read_text_file(Path(req.solution_patch_path))
    desc_text = _read_text_file(Path(req.description_path)) if getattr(req, "description_path", None) else None

    req_obj = {
        "repo_path": req.repo_path,
        "sha": req.sha,
        "dockerfile_text": dockerfile_text,
        "test_patch_text": test_patch_text,
        "solution_patch_text": solution_patch_text,
        "description_text": desc_text,
        "policy": req.policy.model_dump(),
    }

    # Persist inputs as artifacts
    _write_text_file(job_dir / "Dockerfile.problem", dockerfile_text)
    _write_text_file(job_dir / "test.patch", test_patch_text)
    _write_text_file(job_dir / "solution.patch", solution_patch_text)
    if desc_text is not None:
        _write_text_file(job_dir / "description.txt", desc_text)

    _submit_job(job_id, req_obj)

    effective_wait = wait or bool(req.wait)
    effective_timeout = req.wait_timeout_sec if req.wait_timeout_sec is not None else wait_timeout_sec

    if effective_wait:
        try:
            _wait_for_job(job_id, float(effective_timeout))
        except TimeoutError:
            st = _read_status(job_id)
            st["wait"] = {"completed": False, "timeout_sec": effective_timeout}
            return st

    return _read_status(job_id)


@app.get("/v1/jobs/{job_id}")
def job_status(job_id: str) -> Dict[str, Any]:
    # Must never 500.
    try:
        return _read_status(job_id)
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"job_id": job_id, "status": "failed", "reason": e.detail})  # type: ignore[return-value]
    except Exception as e:
        return {"job_id": job_id, "status": "failed", "reason": f"{type(e).__name__}: {e}", "crash_log": "crash.log"}


@app.get("/v1/jobs/{job_id}/status")
def job_status_alias(job_id: str) -> Dict[str, Any]:
    return job_status(job_id)


@app.get("/v1/jobs/{job_id}/summary")
def job_summary(job_id: str) -> Dict[str, Any]:
    st = job_status(job_id)

    status = st.get("status")
    if status == "queued":
        phase = "queued"
    elif status == "running":
        phase = "triad"
    elif status in {"succeeded", "failed"}:
        phase = "done"
    else:
        phase = "unknown"

    code = None
    if status == "succeeded":
        code = 0
    elif status == "failed":
        code = 1

    # Pointers to common artifacts/logs (present when available).
    job_dir = _job_dir(str(job_id))
    pointers = {}
    for name in (
        "crash.log",
        "docker_build.log",
        "git_clone.log",
        "git_checkout.log",
        "triad.log",
        "runner.log",
        "static_postchecks.json",
        "report.json",
    ):
        if (job_dir / name).exists():
            pointers[name.replace('.', '_').replace('-', '_')] = name

    st["phase"] = phase
    st["code"] = code
    st["pointers"] = pointers

    # Surface AI heuristic details (if present) in summary.
    try:
        rep = _json_load(job_dir / "report.json") or {}
        details = (rep.get("AI Difficulty Estimate") or {}).get("Details") or {}
        payload = details.get("ai") or details
        if isinstance(payload, dict):
            st["ai"] = payload
    except Exception:
        pass

    return st


@app.get("/v1/jobs/{job_id}/report")
def job_report(job_id: str) -> Dict[str, Any]:
    job_dir = _job_dir(job_id)
    path = job_dir / "report.json"
    if not path.exists():
        # If job isn't done yet, return a best-effort placeholder with current status.
        st = _read_status(job_id)
        return {
            "Static Checks Summary": {
                "Description": "Not yet available (job still running or report not generated).",
                "Message": "Not available",
                "Details": {"status": st.get("status")},
                "Severity": "INFO",
            }
        }
    return _json_load(path)


@app.get("/v1/jobs/{job_id}/artifacts")
def job_artifacts(job_id: str) -> Dict[str, Any]:
    job_dir = _job_dir(job_id)
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="job not found")
    items: List[Dict[str, Any]] = []
    for p in sorted(job_dir.iterdir(), key=lambda x: x.name):
        if p.is_file():
            st = p.stat()
            items.append({"name": p.name, "size": st.st_size, "mtime": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat().replace("+00:00", "Z")})
    return {"job_id": job_id, "artifacts": items}


@app.get("/v1/jobs/{job_id}/artifacts/{name}")
def job_artifact_meta(job_id: str, name: str) -> Dict[str, Any]:
    name = _safe_name(name)
    job_dir = _job_dir(job_id)
    p = job_dir / name
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="artifact not found")
    st = p.stat()
    preview = ""
    if st.st_size <= 128_000:
        try:
            preview = p.read_text(encoding="utf-8")
        except Exception:
            preview = ""
    return {"job_id": job_id, "name": name, "size": st.st_size, "preview": preview}


@app.get("/v1/jobs/{job_id}/artifacts/{name}/raw")
def job_artifact_raw(job_id: str, name: str) -> Response:
    name = _safe_name(name)
    job_dir = _job_dir(job_id)
    p = job_dir / name
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="artifact not found")
    # Text by default; FileResponse handles both.
    return FileResponse(p)


@app.get("/v1/jobs/{job_id}/artifacts/{name}/tail")
def job_artifact_tail(job_id: str, name: str, lines: int = Query(200, ge=1, le=5000)) -> Dict[str, Any]:
    name = _safe_name(name)
    job_dir = _job_dir(job_id)
    p = job_dir / name
    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="artifact not found")
    try:
        with p.open("r", encoding="utf-8", errors="replace") as f:
            dq: Deque[str] = deque(maxlen=lines)
            for ln in f:
                dq.append(ln.rstrip("\n"))
        return {"job_id": job_id, "name": name, "lines": lines, "tail": "\n".join(dq)}
    except Exception as e:
        return {"job_id": job_id, "name": name, "error": f"{type(e).__name__}: {e}", "tail": ""}

@app.post("/v1/postchecks/ai")
def postchecks_ai(req: StaticInlineRequest) -> Dict[str, Any]:
    return _estimate_ai_pass_rate_heuristic(
        description_text=req.description_text,
        test_patch_text=req.test_patch_text,
        policy=req.policy,
    )

@app.post("/v1/postchecks/ai/from-files")
def postchecks_ai_from_files(req: StaticFromFilesRequest) -> Dict[str, Any]:
    test_patch_text = _read_text_file(Path(req.test_patch_path))
    desc_text = _read_text_file(Path(req.description_path)) if getattr(req, "description_path", None) else None
    return _estimate_ai_pass_rate_heuristic(
        description_text=desc_text,
        test_patch_text=test_patch_text,
        policy=req.policy,
    )


def _ai_simulate_trials(*, pass_rate_0_1: float, key: str, trials: int = 5, seed: int | None = None) -> dict:
    """
    Deterministic low-variance trial simulation.

    Instead of N independent RNG draws (high variance for small N), we use a
    stratified scheme that still reports N "trials" but produces a stable count
    close to pass_rate_0_1 * trials.

    For a fixed (key, seed, trials), passes will be either floor(p*N) or ceil(p*N).
    """
    import hashlib

    try:
        p = float(pass_rate_0_1)
    except Exception:
        p = 0.0
    p = max(0.0, min(1.0, p))

    try:
        n = int(trials)
    except Exception:
        n = 5
    if n <= 0:
        n = 1

    payload = (key or "").encode("utf-8", errors="replace")
    if seed is not None:
        payload += b"\n" + str(int(seed)).encode("ascii", errors="ignore")

    digest = hashlib.blake2b(payload, digest_size=8).digest()
    u0 = int.from_bytes(digest, "big") / 2**64  # [0,1)

    passes = 0
    for i in range(n):
        # stratified thresholds: (i+u0)/n
        if ((i + u0) / n) < p:
            passes += 1

    return {"trials": n, "passes": passes, "fails": n - passes, "seed": int.from_bytes(digest, "big")}
def _ai_score_to_pass_rate(hardness_score: float, policy: Any) -> float:
    """Map hardness score (0..100) to a tunable predicted AI pass rate (0..1)."""
    try:
        center = float(getattr(policy, "ai_curve_center", 72.0))
        slope = float(getattr(policy, "ai_curve_slope", 10.0))
    except Exception:
        center, slope = 72.0, 10.0

    if slope <= 0:
        slope = 10.0

    try:
        x = (float(hardness_score) - center) / slope
        return 1.0 / (1.0 + math.exp(x))
    except OverflowError:
        # very large x -> 0.0, very negative x -> 1.0
        return 0.0 if float(hardness_score) > center else 1.0

