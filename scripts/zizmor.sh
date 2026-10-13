#!/usr/bin/env bash
# Online audits need a GitHub token; without one zizmor skips them.
set -euo pipefail

token="${GH_TOKEN:-${GITHUB_TOKEN:-}}"
if [ -z "$token" ] && command -v gh >/dev/null 2>&1; then
    token="$(gh auth token 2>/dev/null || true)"
fi

if [ -n "$token" ]; then
    export GH_TOKEN="$token"
else
    # An empty GH_TOKEN is an error to zizmor, not "no token".
    unset GH_TOKEN
    echo "zizmor: no GitHub token (set GH_TOKEN/GITHUB_TOKEN or run 'gh auth" \
        "login'); online audits are skipped locally — CI runs them with a token." >&2
fi

exec uvx zizmor@1.30.1 .github/
