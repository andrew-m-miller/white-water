#!/usr/bin/env bash
#
# Fails if a binary requires a newer glibc than Flame's certified Linux provides.
#
# White Water targets Rocky Linux 9.5+, which ships glibc 2.34. A plugin built on a modern
# distribution silently picks up newer symbol versions and then refuses to load on the
# Flame box with an error that names a symbol, not the real cause. Worse, the host usually
# reports nothing at all -- the plugin is simply absent from the menu.
#
# Run this on every Linux artifact before it is handed to anyone.
#
#   scripts/check-glibc-baseline.sh <binary> [max glibc version]

set -euo pipefail

binary="${1:-}"
max_version="${2:-2.34}"

if [[ -z "$binary" ]]; then
  echo "usage: $0 <binary> [max glibc version]" >&2
  exit 2
fi

if [[ ! -f "$binary" ]]; then
  echo "not found: $binary" >&2
  exit 2
fi

# Sorts versions numerically so 2.9 does not compare as newer than 2.28.
version_gt() {
  [[ "$1" != "$2" ]] && [[ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | tail -1)" == "$1" ]]
}

mapfile -t required < <(
  objdump -T "$binary" 2>/dev/null |
    grep -o 'GLIBC_[0-9][0-9.]*' |
    sed 's/GLIBC_//' |
    sort -uV
)

if [[ ${#required[@]} -eq 0 ]]; then
  echo "$binary references no versioned glibc symbols"
  exit 0
fi

highest="${required[-1]}"

echo "glibc symbol versions required by $(basename "$binary"):"
printf '  %s\n' "${required[@]}"
echo "highest: $highest (baseline: $max_version)"

if version_gt "$highest" "$max_version"; then
  echo
  echo "FAIL: needs glibc $highest but the baseline is $max_version." >&2
  echo "This binary will not load on the Rocky 9 box Flame runs on." >&2
  echo "Build it in a rockylinux:9 container -- see .github/workflows/ci.yml." >&2
  exit 1
fi

echo "OK: within the baseline."
