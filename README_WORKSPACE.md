# Offline Codex Workspace Template

This zip is meant to be extracted into the task root as `/workspace`.
It contains `validator-local-codex/` (validator runner + scripts) and empty bundle locations.

## What you must add for each problem
Codex cannot fetch from the internet. For a given `Repo URL` + `PINNED_SHA`, you MUST provide an offline git source that contains that commit.
Recommended: a git bundle named after the repo, stored at:

- `/workspace/validator-local-codex/bundles/<repo>.bundle`
  (example: `arrow.bundle` for `https://github.com/arrow-py/arrow`)

The scripts can also find bundles under `/workspace/bundles/`.

### Create a bundle (run on any machine that has the repo)
If you already have a local clone:

- `cd /path/to/repo`
- `git bundle create <repo>.bundle --all`

Or if you want only the pinned commit and its history:

- `git bundle create <repo>.bundle <PINNED_SHA> --all`

Then copy `<repo>.bundle` into one of the bundle folders above.

### Quick verification
After copying the bundle into the workspace:

- `git clone /workspace/validator-local-codex/bundles/<repo>.bundle /workspace/repo`
- `git -C /workspace/repo cat-file -e <PINNED_SHA>^{commit}`

If that succeeds, Codex can reproduce and iterate.

## Typical Codex run sequence
1) Start validator services (if present):
   - `cd /workspace/validator-local-codex`
   - `bash scripts/task_bootstrap.sh`

2) Find/clone the repo offline:
   - Set `REPO_URL` to the repo URL.
   - Run: `python3 /workspace/validator-local-codex/scripts/task_find_repo.py`

3) Put the submission files under `<REPO_DIR>/submission/`:
   - `Dockerfile.problem`, `test.patch`, `solution.patch`, `description.txt`, `pinned_sha.txt`

4) Build and reproduce inside Docker using `Dockerfile.problem`.

## Notes
- Nothing here is “stealing” a repo. A bundle is just an offline transport of a git repository you already have access to.
- You do NOT need to include every repo in the workspace, only the ones you plan to validate.


## Codex setup script (recommended)

In Codex "Setup script", run:

- bash scripts/codex_setup.sh

This will:
- symlink /workspace/validator-local-codex to the bundled validator runner
- symlink any ./bundles/*.bundle to /workspace/*.bundle
- optionally fetch a bundle during setup if you set env vars:
  - BUNDLE_REPO_URL (required to fetch)
  - PINNED_SHA (optional validation)
  - BUNDLE_NAME (optional; defaults to repo name)

## Docker requirement

If your task prompt uses `docker build` / `docker run`, make sure your Codex container image includes a working `docker` CLI.
Verify in terminal: `command -v docker && docker --version`
