# Run checklist (any repo)

Before calling something "final", confirm:

- Repo is at PINNED_SHA and clean (no fetching):
  - git cat-file -e SHA^{commit}
  - git checkout --detach SHA
  - git status --porcelain

- validator-local:
  - preflight PASS or only expected warnings
  - triad PASS

- Manual triad from clean tree:
  - apply test.patch -> ./test.sh base PASS
  - apply test.patch -> ./test.sh new FAIL
  - apply solution.patch -> ./test.sh new PASS
  - apply solution.patch -> ./test.sh base PASS

- Patch boundaries:
  - test.patch touches only test.sh + one tests/test_*_problem.py
  - solution.patch touches only production files

- description.txt:
  - ASCII-only
  - 70-88 words
  - documents every behavior enforced by tests
  - no headings, no test file mentions, no shell commands
