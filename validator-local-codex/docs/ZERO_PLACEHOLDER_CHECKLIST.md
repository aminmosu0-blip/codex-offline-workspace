# Zero-placeholder setup checklist

To prevent Codex from stopping on placeholders:

1) Launch Codex from inside the target repo:
   cd /abs/path/to/repo
   codex

2) Put the 4 artifacts in:
   /abs/path/to/repo/submission/
   - Dockerfile.problem
   - test.patch
   - solution.patch
   - description.txt
   Optional:
   - pinned_sha.txt  (single SHA token)

3) Ensure validator-local server is running:
   uvicorn app:app --host 127.0.0.1 --port 8000

4) Ensure Codex can find vlrun.py:
   - default: ~/validator-local-codex/vlrun.py
   - or set: export VLRUN=/abs/path/to/vlrun.py

Then paste:
   prompts/REVISION_AUTODETECT_PROMPT.txt
