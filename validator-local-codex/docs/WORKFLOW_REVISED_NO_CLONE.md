# Revised-problem workflow (no cloning)

This workflow assumes the user provides:
- REPO_DIR: local repo working tree already on disk
- PINNED_SHA: commit SHA to validate against
- SUB_DIR: Dockerfile.problem, test.patch, solution.patch, description.txt
- Validator output + reviewer notes

1) Verify repo state
   - cd REPO_DIR
   - git rev-parse HEAD (must equal PINNED_SHA)
   - git status --porcelain (should be clean)

2) Reproduce the triad manually (fast sanity)
   - git reset --hard
   - git clean -xdf
   - git apply SUB_DIR/test.patch
   - ./test.sh base  (must PASS)
   - ./test.sh new   (must FAIL)
   - git apply SUB_DIR/solution.patch
   - ./test.sh new   (must PASS)
   - ./test.sh base  (must PASS)

3) Run validator-local (source of truth)
   - python3 vlrun.py preflight --base-url http://127.0.0.1:8000 --repo REPO_DIR --sha PINNED_SHA --dockerfile SUB_DIR/Dockerfile.problem --test-patch SUB_DIR/test.patch --solution-patch SUB_DIR/solution.patch --description SUB_DIR/description.txt
   - python3 vlrun.py triad    --base-url http://127.0.0.1:8000 --repo REPO_DIR --sha PINNED_SHA --dockerfile SUB_DIR/Dockerfile.problem --test-patch SUB_DIR/test.patch --solution-patch SUB_DIR/solution.patch --description SUB_DIR/description.txt

4) Revise in this order
   a) description alignment (ASCII-only, 70-88 words, no undocumented requirements)
   b) tests (fair hardening, remove brittleness, keep patch boundaries)
   c) solution (production-only)

5) Regenerate patches from staged index only (never edit patch files directly)

Repeat until validator PASS.
