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

if [ -n "$token" ]; then
    export GH_TOKEN="$token"
else
    # An *empty* GH_TOKEN is an error to zizmor ("token cannot be empty"), not
    # "no token", so unset it and let zizmor run its offline audits only. The
    # dedicated CI zizmor job runs the online audits with a real token.
    unset GH_TOKEN
    echo "zizmor: no GitHub token (set GH_TOKEN/GITHUB_TOKEN or run 'gh auth" \
        "login'); online audits are skipped locally — CI runs them with a token." >&2
fi

exec uvx zizmor@1.26.1 .github/
