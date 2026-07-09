#!/usr/bin/env bash
# Audit the GitHub Actions workflows with zizmor, matching the CI job so a
# finding surfaces at pre-push rather than CI-only.
#
# zizmor's online audits (e.g. ref-version-mismatch — a hash pin whose version
# comment has drifted from the tag it now resolves to) need a GitHub token;
# without one they are silently skipped. Supply a token from the environment or
# the gh CLI so the local run sees the same findings CI does. Pinned to the same
# version as .github/workflows/test.yml.
set -euo pipefail

token="${GH_TOKEN:-${GITHUB_TOKEN:-}}"
if [ -z "$token" ] && command -v gh >/dev/null 2>&1; then
    token="$(gh auth token 2>/dev/null || true)"
fi
if [ -z "$token" ]; then
    echo "zizmor: no GitHub token (set GH_TOKEN/GITHUB_TOKEN or run 'gh auth" \
        "login'); online audits will be skipped — CI runs them with a token." >&2
fi

GH_TOKEN="$token" exec uvx zizmor@1.26.1 .github/
