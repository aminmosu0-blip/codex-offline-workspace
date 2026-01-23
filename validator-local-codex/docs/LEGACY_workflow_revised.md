# Codex + validator-local (from-files) workflow for REVISED problems (repo URL + pinned SHA)

This version matches your updated validator-local API (`/v1/jobs/from-files`) and the bundled client `vlrun.py`.
It does **not** require repo.zip.

## What Codex must produce at the end (the 4 artifacts)

In one submission folder, produce exactly:

1) `Dockerfile.problem`
2) `test.patch`
3) `solution.patch`
4) `description.txt`

Everything else (logs, novelty_gate.txt, notes.txt) is optional and should not be required by the platform.

---

## 0) One-time setup: validator-local (server + client)

### Install deps (local venv recommended)
```bash
set -euo pipefail
cd ~/validator-local
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r requirements.txt
```

### Run the server
```bash
. ~/validator-local/.venv/bin/activate
uvicorn app:app --host 127.0.0.1 --port 8000
```

### Client usage (stdlib)
From the validator-local folder:
```bash
python3 vlrun.py --help
python3 vlrun.py preflight --help
python3 vlrun.py triad --help
```

---

## 1) Start a fresh revision run (host)

```bash
set -euo pipefail
RUN="$HOME/rev-run-$(date +%Y%m%d_%H%M%S)"
REPO="$RUN/repo"
SUB="$RUN/submission"
mkdir -p "$REPO" "$SUB"
echo "RUN=$RUN"
```

You will place (or update) the 4 artifacts inside `$SUB/`.

---

## 2) Clone the real repo and pin SHA (host)

Inputs you provide:
- `REPO_URL`
- `PINNED_SHA`

```bash
set -euo pipefail
cd "$RUN"
git clone "$REPO_URL" "$REPO"
cd "$REPO"
git fetch --all --tags --prune
git checkout "$PINNED_SHA"
git rev-parse HEAD
git status -sb
```

If checkout fails, stop: the SHA is wrong or not reachable.

---

## 3) Put the current artifacts into the submission folder (host)

Required files:
- `$SUB/Dockerfile.problem`
- `$SUB/test.patch`
- `$SUB/solution.patch`
- `$SUB/description.txt`

Optional:
- `$SUB/reviewer_notes.txt`
- `$SUB/validator_notes.txt`

---

## 4) Validate quickly (preflight) and run the triad (host)

Use absolute paths.

```bash
set -euo pipefail
BASE_URL="http://127.0.0.1:8000"

python3 ~/validator-local/vlrun.py preflight   --base-url "$BASE_URL"   --repo "$REPO"   --sha "$PINNED_SHA"   --dockerfile "$SUB/Dockerfile.problem"   --test-patch "$SUB/test.patch"   --solution-patch "$SUB/solution.patch"   --description "$SUB/description.txt" || true

python3 ~/validator-local/vlrun.py triad   --base-url "$BASE_URL"   --repo "$REPO"   --sha "$PINNED_SHA"   --dockerfile "$SUB/Dockerfile.problem"   --test-patch "$SUB/test.patch"   --solution-patch "$SUB/solution.patch"   --description "$SUB/description.txt" || true
```

The triad is:
- apply `test.patch`, run `./test.sh base` (must pass)
- run `./test.sh new` on baseline (must fail)
- apply `solution.patch`, run base + new (both must pass)

---

## 5) Rules Codex must follow while revising

### Patch boundaries (non-negotiable)
- `test.patch` may change ONLY:
  - `test.sh`
  - exactly one new `tests/test_*_problem.py`
- `solution.patch` may change ONLY production code (no tests, no docs, no tooling)

### Dockerfile.problem rules
- Dockerfile.problem is the only install step (container must run offline after build).
- Do not upgrade pip.
- Keep deps minimal but sufficient for base tests + new tests.

### Tests rules
- Deterministic, offline, ASCII-only.
- Avoid brittle assertions: no dependency/version pins, no exact error strings unless required, no required JSON keys unless documented.
- Harden fairly:
  - >=5 independent invariants
  - >=2 anti-shortcut traps
- Base must ignore the new test file (use `--ignore` in base mode).
- New mode must run ONLY the new test file.

### Description rules
- ASCII-only, 70-88 words, no headings, no test mentions, no shell commands, no solution hints.
- Must document every behavior the tests enforce (no undocumented requirements).

---

## 6) How to regenerate patches correctly (host, from staged index only)

Work inside the pinned repo clone (`$REPO`), and write patch outputs into `$SUB`.

### Regenerate test.patch (tests-only)
```bash
set -euo pipefail
cd "$REPO"
git reset

git add -f test.sh tests/test_<topic>_problem.py
git diff --cached --no-color > "$SUB/test.patch"
git reset

python3 - <<'PY'
from pathlib import Path
p = Path(r"$SUB/test.patch")
t = p.read_text(encoding="utf-8", errors="replace")
bad = ["\r\n", "\t"]
print("test.patch bytes:", p.stat().st_size)
PY
```

### Regenerate solution.patch (production-only)
```bash
set -euo pipefail
cd "$REPO"
git reset

git add -f <production_files_only>
git diff --cached --no-color > "$SUB/solution.patch"
git reset

ls -la "$SUB/solution.patch"
```

---

## 7) Codex prompts (paste into Codex app)

Put an `AGENTS.md` in the repo root or the run folder, then use these prompts.

### REVIEW/REVISE prompt (recommended)
```
Read AGENTS.md and follow it exactly.

We are revising an existing platform problem.

Inputs:
- Repo URL: <REPO_URL>
- Pinned SHA: <PINNED_SHA>
- Submission folder (4 artifacts): <ABS_PATH_TO_SUB>

Required output at end:
Dockerfile.problem, test.patch, solution.patch, description.txt in <SUB>.

Work order:
1) Reproduce triad via:
   python3 ~/validator-local/vlrun.py preflight ... and triad ...
2) Summarize failures and map each to the smallest change needed.
3) Apply reviewer notes first unless they would remove a behavior enforced by tests (then rewrite more compactly).
4) Harden tests fairly (>=5 invariants, >=2 anti-shortcut traps) while keeping patch boundaries.
5) Ensure description.txt matches exactly what tests enforce (ASCII-only, 70-88 words).
6) Regenerate patches from staged index only, rerun triad until PASS.
7) End by listing the 4 artifact paths and their sizes, and show which files each patch touches.
```

### How to show patch touch lists (Codex should do this)
```
python3 - <<'PY'
from pathlib import Path
import re
for name in ("test.patch","solution.patch"):
    p = Path("<SUB>")/name
    files=[]
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("+++ b/"):
            files.append(line[6:])
    print(name, "touches:")
    for f in files:
        print(" -", f)
PY
```
