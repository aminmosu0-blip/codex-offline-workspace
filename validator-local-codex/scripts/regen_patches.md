# Regenerate patches from staged index only

Run these inside REPO_DIR. Write outputs into SUB_DIR.

test.patch (tests only):
  git reset
  git add -f test.sh tests/test_<topic>_problem.py
  git diff --cached --no-color > SUB_DIR/test.patch
  git reset

solution.patch (production only):
  git reset
  git add -f <production_files_only>
  git diff --cached --no-color > SUB_DIR/solution.patch
  git reset

Then re-run:
  ./test.sh base
  ./test.sh new
  python3 vlrun.py triad ...
