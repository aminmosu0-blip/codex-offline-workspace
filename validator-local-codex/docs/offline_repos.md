Offline repos for Codex tasks

This Codex Task workspace runs without network access. To reproduce a submission against a pinned commit, you must
provide the target git repository locally.

Preferred layout:
  /workspace/validator-local-codex/   (this tool)
  /workspace/repo/                   (materialized working tree, created by cloning a bundle/mirror)
  /workspace/validator-local-codex/bundles/<name>.bundle  (input bundle or bare mirror)

Create a bundle (full history):
  git clone --mirror <REPO_URL> <name>.git
  git -C <name>.git bundle create <name>.bundle --all

Create a smaller bundle with only one pinned commit (still clonable):
  git clone <REPO_URL> <name>
  git -C <name> branch pinned <PINNED_SHA>
  git -C <name> bundle create <name>.bundle pinned

Verification:
  git -C /workspace/repo cat-file -e <PINNED_SHA>^{commit}

Notes:
- Bundles and bare mirrors are local files, so they work offline.
- Do not use shallow clones unless you are sure the pinned commit is included.
