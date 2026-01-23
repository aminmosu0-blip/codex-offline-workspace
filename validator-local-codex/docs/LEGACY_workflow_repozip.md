# Codex + validator-local: no-headache platform-problem workflow (host-driven)

This is tailored to the two ZIPs you uploaded:

- `repo.zip` (target repo): `new-main.zip` (folder name `new-main/`)
- `validator-local.zip` (validator runner): `validator-local-main.zip` (folder name `validator-local-main/`)

Note: both ZIPs appear to be the *same project family* ("Validator Local (Milestone A)"); `new-main.zip` includes extra docs and `pydantic_settings/`. Treat `new-main.zip` as the *target repo* and `validator-local-main.zip` as the *validator tool*.

## 0) One-time: install validator-local as a CLI (recommended)

```bash
set -euo pipefail
cd ~
rm -rf ~/validator-local
mkdir -p ~/validator-local
cd ~/validator-local
unzip -q /ABS/PATH/TO/validator-local-main.zip

# normalize to the extracted top-level directory
cd validator-local-main

python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -e .
validator --help
```

You can now validate any submission folder with:
```bash
validator static --dir /ABS/PATH/TO/submission
validator triad  --dir /ABS/PATH/TO/submission
```

Validator outputs are written under:
`/ABS/PATH/TO/submission/.validator_runs/<job_id>/` (report.json, triad_summary.json, bundle.zip, stage_logs/*)

## 1) Start a new run (fresh workspace every time)

```bash
set -euo pipefail
RUN="$HOME/triad-run-$(date +%Y%m%d_%H%M%S)"
SUB="$RUN/submission"
DEV="$RUN/dev_repo"
mkdir -p "$SUB" "$DEV"
echo "RUN=$RUN"
```

Copy the target repo zip into submission (the validator auto-detects `repo.zip`):
```bash
cp -f /ABS/PATH/TO/new-main.zip "$SUB/repo.zip"
```

## 2) Create a dev git baseline that matches validator behavior

The validator extracts `repo.zip` and, if it finds exactly one top-level directory, it treats that directory as the repo root and `git init` commits a baseline. Do the same for development so your patches apply cleanly.

```bash
set -euo pipefail
cd "$DEV"
unzip -q "$SUB/repo.zip"

# normalize to single top-level dir if present
ROOT="$DEV"
entries=("$DEV"/*)
if [ "${#entries[@]}" -eq 1 ] && [ -d "${entries[0]}" ]; then
  ROOT="${entries[0]}"
fi
echo "REPO_ROOT=$ROOT"

cd "$ROOT"
git init
git add -A
git commit -m baseline
```

From here on, do ALL edits in `$ROOT` and generate patches into `$SUB`.

## 3) Minimal Dockerfile.problem (in submission folder, not in repo)

This Dockerfile is used by the validator to build the image. The container runs with `--network none`,
so everything needed must be installed at build time.

Create: `$SUB/Dockerfile.problem`

```Dockerfile
FROM public.ecr.aws/x8v8d7g8/mars-base:latest
WORKDIR /app
COPY . .
RUN python3 -m pip install -e . &&     python3 -m pip install pytest hypothesis
CMD ["/bin/bash"]
```

(If your new tests need extra deps, add them here. Do NOT install anything at runtime.)

## 4) test.sh rules that satisfy validator-local

- must support `base` and `new`
- must contain `--ignore` somewhere (validator checks for this token)
- base should ignore the new test file and may treat "no tests collected" (exit code 5) as success
- new runs ONLY the new test module

Template (edit `<topic>`):

```bash
cat > test.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
mode="${1:-}"
case "$mode" in
  base)
    python3 -m pytest --ignore tests/test_<topic>_problem.py
    rc=$?
    if [ "$rc" -eq 5 ]; then exit 0; fi
    exit "$rc"
    ;;
  new)
    python3 -m pytest tests/test_<topic>_problem.py
    ;;
  *)
    echo "usage: $0 {base|new}" >&2
    exit 2
    ;;
esac
EOF
chmod +x test.sh
```

## 5) Build test.patch (tests-only) from staged index

Inside `$ROOT` (repo root):

```bash
set -euo pipefail
cd "$ROOT"

# add/modify ONLY these two files for test.patch:
# - test.sh
# - tests/test_<topic>_problem.py
git reset
git add -f test.sh tests/test_<topic>_problem.py
git diff --cached --no-color > "$SUB/test.patch"
git reset
ls -la "$SUB/test.patch"
```

## 6) Build solution.patch (production-only) from staged index

After freezing tests, change production code only, then:

```bash
set -euo pipefail
cd "$ROOT"
git reset
git add -f <production_files_only>
git diff --cached --no-color > "$SUB/solution.patch"
git reset
ls -la "$SUB/solution.patch"
```

## 7) description.txt (ASCII-only, 70-88 words, no headings)

Write to: `$SUB/description.txt`

Checklist:
- ASCII only (no curly quotes)
- 70-88 words
- no section labels/headings
- no test mentions, no shell commands, no fix hints
- documents EVERY behavior enforced by the new tests

## 8) Validate (static + triad)

```bash
set -euo pipefail
validator static --dir "$SUB" || true
validator triad  --dir "$SUB" || true
```

Then inspect the newest run:
```bash
ls -lat "$SUB/.validator_runs" | head
# open report:
ls -lat "$SUB/.validator_runs"/*/report.json | head -n 1
```

## 9) Codex app prompts (copy/paste)

### New problem kickoff prompt (Codex)

Paste into Codex (running in `$ROOT` directory):

```
Read AGENTS.md and follow it exactly.

We are authoring a platform-style problem. Work host-driven:
- Dev repo root is the current directory (a git baseline commit exists).
- Submission folder is: <ABS_PATH_TO_SUB>
- Validator CLI is available as: validator

Work order:
1) Propose 2-3 candidate issues (module/function + symptom) that are deterministic and offline.
2) For each candidate, run an upstream novelty gate: repo issue/PR search + web search. If any plausible match, discard.
3) Pick 1 'Novelty Gate: PASS' idea, write /work/novelty_gate.txt into the submission folder.
4) Write test.sh + exactly one tests/test_<topic>_problem.py (ASCII-only, no comments, >=5 invariants, >=2 anti-shortcut traps).
5) Generate <SUB>/test.patch from staged index only, then run: validator triad --dir <SUB>.
6) Freeze tests, then fix production code only, generate <SUB>/solution.patch, rerun validator triad.
7) Write description.txt (ASCII-only, 70-88 words) documenting every enforced behavior.
End with exactly 4 artifacts present in <SUB>: Dockerfile.problem, test.patch, solution.patch, description.txt.
```

### Review/revise prompt (Codex)

```
Read AGENTS.md and follow it exactly.

We are revising an existing platform problem. Submission folder: <ABS_PATH_TO_SUB>.
I will paste validator output and reviewer notes.

First:
- Run validator triad --dir <SUB> and summarize failures/mismatches.
- Propose a minimal fix plan (what files to touch and why), then implement.
- Regenerate patches from staged index only and re-run triad until PASS.
- Ensure the final 4 artifacts exist in <SUB>.
```
