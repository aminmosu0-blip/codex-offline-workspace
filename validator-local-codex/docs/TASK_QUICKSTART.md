# Codex Task quickstart (no placeholder stalls)

1) Start validator-local inside the task:
   bash scripts/task_bootstrap.sh

2) In a separate terminal (or after bootstrap), locate the target repo:
   export REPO_DIR="$(python3 scripts/task_find_repo.py)"

   If REPO_DIR is empty, set it explicitly:
   export REPO_DIR=/workspace/your-repo

3) Ensure the 4 artifacts exist at:
   $REPO_DIR/submission/
     Dockerfile.problem
     test.patch
     solution.patch
     description.txt
   Optional:
     pinned_sha.txt

4) Paste prompts/TASK_REVISION_PROMPT.txt into Codex.
