# validator-local (Codex workflow repo)

This repo is a local validator service + a small client (vlrun.py). It is designed so Codex can revise or author platform-style Python problems for ANY target repository, using only:
- a local repo directory path (REPO_DIR)
- a pinned commit SHA (PINNED_SHA)
- a submission folder containing artifacts (SUB_DIR)
- pasted validator output + reviewer notes

Codex default: no cloning, no fetching, no web browsing. Work only with what the user provides on disk.

Quickstart (host)
1) Create venv + install:
   python3 -m venv .venv
   . .venv/bin/activate
   python3 -m pip install -r requirements.txt

2) Run the server:
   uvicorn app:app --host 127.0.0.1 --port 8000

3) Run preflight + triad (absolute paths):
   BASE_URL=http://127.0.0.1:8000
   REPO_DIR=/ABS/REPO_DIR
   PINNED_SHA=<PINNED_SHA>
   SUB_DIR=/ABS/SUB_DIR
   python3 vlrun.py preflight --base-url "$BASE_URL" --repo "$REPO_DIR" --sha "$PINNED_SHA" --dockerfile "$SUB_DIR/Dockerfile.problem" --test-patch "$SUB_DIR/test.patch" --solution-patch "$SUB_DIR/solution.patch" --description "$SUB_DIR/description.txt"
   python3 vlrun.py triad    --base-url "$BASE_URL" --repo "$REPO_DIR" --sha "$PINNED_SHA" --dockerfile "$SUB_DIR/Dockerfile.problem" --test-patch "$SUB_DIR/test.patch" --solution-patch "$SUB_DIR/solution.patch" --description "$SUB_DIR/description.txt"

Prompts
- prompts/REVISION_MASTER_PROMPT.txt: use when you already have Dockerfile.problem + test.patch + solution.patch + description.txt and need to iterate.
- prompts/NEW_PROBLEM_MASTER_PROMPT.txt: use when you are authoring a new problem from scratch in an existing local repo directory (still no cloning by default).

AGENTS.md is the source of truth for Codex behavior.

---

Original README content (preserved)

validator-local
==============

A small local FastAPI service that validates "platform-style" Python problem artifacts against a repo checkout.
It runs deterministic static checks and (optionally) the full triad:

- apply test.patch and run `./test.sh base` (must pass)
- run `./test.sh new` on base (must fail)
- apply solution.patch and run `./test.sh base` + `./test.sh new` (both must pass)

Quick start
-----------

1) Create a local virtual environment (recommended, but not required):

   - Windows PowerShell:
     - `py -m venv .venv`
     - `.\.venv\Scripts\Activate.ps1`
     - `pip install -r requirements.txt`

   - Linux/WSL:
     - `python3 -m venv .venv`
     - `source .venv/bin/activate`
     - `pip install -r requirements.txt`

2) Run the server:

   - `uvicorn app:app --host 127.0.0.1 --port 8000`

3) Run a preflight (fast-fail) + triad using the bundled client:

   - `python vlrun.py preflight --repo <path> --sha <sha-or-HEAD> --dockerfile <Dockerfile.problem> --test-patch <test.patch> --solution-patch <solution.patch>`
   - `python vlrun.py triad     --repo <path> --sha <sha-or-HEAD> --dockerfile <Dockerfile.problem> --test-patch <test.patch> --solution-patch <solution.patch>`

Notes
-----

- You do NOT need to ship a `.venv` inside this folder/zip. Keep the venv local.
- The triad uses Docker when available. On Windows hosts, user remapping (`-u uid:gid`) is skipped for compatibility.


Codex Task quickstart
- bash scripts/task_bootstrap.sh  (starts validator on 127.0.0.1:8000 and prints exports)
- docs/TASK_QUICKSTART.md has the no-placeholder workflow
