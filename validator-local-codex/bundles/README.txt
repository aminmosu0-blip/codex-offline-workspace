Offline repo bundles for Codex tasks

This workspace is offline. To let Codex reproduce and revise problems, provide the target repo as a local git bundle
or a local bare mirror.

Recommended (full history):
  git clone --mirror <REPO_URL> <name>.git
  git -C <name>.git bundle create <name>.bundle --all

Smaller (only one pinned commit, still clonable):
  git clone <REPO_URL> <name>
  git -C <name> branch pinned <PINNED_SHA>
  git -C <name> bundle create <name>.bundle pinned

Place one of these files here:
  bundles/<name>.bundle
  bundles/<name>.git

Codex helper scripts will clone it to /workspace/repo when /workspace/repo is missing.
