# AGENTS.md

You are Codex running on the host.

Mission
- Revise an existing platform-style Python problem until validator + reviewer checks pass.
- Default mode is "no cloning": work only with the repo directory and artifacts the user provides.

Inputs the user provides for each run
- REPO_DIR: absolute path to the local repo working tree (already present on disk)
- PINNED_SHA: the commit SHA the run must validate against
- SUB_DIR: absolute path to the submission folder containing:
  - Dockerfile.problem
  - test.patch
  - solution.patch
  - description.txt
- Validator output + reviewer notes

Non-negotiable deliverables at the end
- Final Title text
- Final description.txt (ASCII-only, 70-88 words, no headings, no test mentions, no shell commands)
- Final test.patch
- Final solution.patch

Global hard rules (do not violate)
- Do NOT clone repos or fetch network resources unless the user explicitly asks.
- Tests must be deterministic and offline: no network, no sleeps, no wall-clock dependence.
- ASCII-only for problem text and tests: UTF-8, LF-only, straight quotes (no smart punctuation).
- Use public APIs only in tests. Avoid private internals and brittle exact-string assertions unless required.
- Keep it fast: aim for comfortably under 3 minutes total runtime.

Patch boundaries (strict)
- test.patch may modify ONLY:
  - test.sh
  - exactly one new test file directly under tests/: tests/test_*_problem.py
  - (rare) minimal tests/conftest.py only if unavoidable and repo-compatible
- solution.patch may modify ONLY production code (no tests, no tooling, no docs, no Dockerfile).

test.sh conventions
- Exactly two modes: base and new.
- base runs the upstream suite while excluding the new test file and must PASS.
- new runs ONLY the new test file and must FAIL on the base commit.
- Avoid AI-looking wrappers (no piping to tail). Prefer running pytest normally.
- If upstream pytest addopts break isolation, use -o addopts= in new.
- If needed for permissions, add: -o cache_dir=/tmp/pytest-cache
- Recommended env in test.sh: PYTHONNOUSERSITE=1, TZ=UTC, LC_ALL=C.UTF-8, LANG=C.UTF-8

Required work order (never skip)
1) Reproduce triad locally (fast sanity)
   - From a clean repo tree:
     git reset --hard
     git clean -xdf
     git apply SUB_DIR/test.patch
     ./test.sh base  (must PASS)
     ./test.sh new   (must FAIL)
     git apply SUB_DIR/solution.patch
     ./test.sh new   (must PASS)
     ./test.sh base  (must PASS)

2) Reproduce validator-local checks using absolute paths
   - python3 vlrun.py preflight ...
   - python3 vlrun.py triad ...

3) Apply reviewer notes first
   - If a reviewer requests removing text that documents an enforced behavior, rewrite it more compactly instead.

4) If AI pass rate is too high, harden tests fairly
   - Add >=5 independent invariants
   - Add >=2 anti-shortcut traps (e.g., coercion traps, multi-entrypoint coverage)
   - Do not introduce undocumented requirements

5) Regenerate patches from staged index only (never hand-edit patch files)
   - git add ... ; git diff --cached --no-color > SUB_DIR/test.patch (or solution.patch)

End-of-run report (print these)
- Final Title
- Final description.txt content
- Patch touch lists (files changed in each patch)
- Sizes of the four artifacts

Helpers in this repo
- scripts/patch_touches.py shows which files a patch modifies.
- scripts/regen_patches.md shows patch regeneration commands.

Autodetect to avoid placeholder stalls
- If the user prompt contains placeholders for REPO_DIR/SUB_DIR/VLRUN, do not stop immediately.
  First, try to infer:
  - REPO_DIR from `git rev-parse --show-toplevel` using the current working directory.
  - SUB_DIR from `REPO_DIR/submission` or a folder containing Dockerfile.problem under REPO_DIR.
  - PINNED_SHA from `SUB_DIR/pinned_sha.txt` if present, otherwise ask the user for a SHA only after listing what you tried.
  - vlrun.py from $VLRUN or ~/validator-local-codex/vlrun.py.
- If any required input is still missing, ask a single compact question that lists:
  what you found, what is missing, and the exact command(s) the user can run to fix it.


Codex Task mode
- If running in a hosted Codex Task workspace, do not assume validator-local is running externally.
  Start it inside the workspace with: bash scripts/task_bootstrap.sh
- If REPO_DIR is not provided, attempt to discover a likely target repo with:
  python3 scripts/task_find_repo.py
  If discovery fails, ask for REPO_DIR as a single absolute path.



Offline repo bundles
- If the target repo is missing, prefer cloning from a local bundle/mirror under validator-local-codex/bundles/.
- task_find_repo.py can materialize /workspace/repo from a local <name>.bundle or <name>.git when REPO_URL and PINNED_SHA are set.
