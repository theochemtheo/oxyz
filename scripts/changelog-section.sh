#!/usr/bin/env bash
# Print the CHANGELOG.md body for one version — the lines between its heading
# and the next version heading, used as the GitHub Release notes. Run it to
# preview what a tag will publish:
#
#     scripts/changelog-section.sh 0.2.0
#
set -euo pipefail

version="${1:?usage: changelog-section.sh <version>}"
changelog="${2:-CHANGELOG.md}"

section=$(awk -v ver="$version" '
  $0 ~ "^## \\[" ver "\\]"      { grab = 1; next }
  grab && /^## \[/              { exit }
  grab && /^\[[^]]+\]: /        { exit }   # link-reference block at the file foot
  grab {
    if (!started && $0 == "") next   # skip blank lines after the heading
    started = 1
    line[++n] = $0
  }
  END {
    while (n > 0 && line[n] == "") n--   # drop trailing blank lines
    for (i = 1; i <= n; i++) print line[i]
  }
' "$changelog")

if [ -z "$section" ]; then
  echo "no CHANGELOG section for version $version" >&2
  exit 1
fi

printf '%s\n' "$section"
