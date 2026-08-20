#!/usr/bin/env bash
#
# Fails if a binary requires a newer glibc than Flame's certified Linux provides.
#
# A plugin built on a newer distribution silently picks up newer symbol versions and then
# refuses to load on the Flame box, reporting a symbol rather than the real cause -- and
# often reporting nothing at all, the plugin simply being absent from the menu.
#
# Artifacts are therefore built on EL8 (glibc 2.28), so they load on any EL8 or EL9 host
# whatever point release it sits at. A nominal distro version is NOT a glibc version:
# measured 2026-08-20, a Rocky 9.5 build container carried glibc symbols that the certified
# Rocky 9.5 Flame box did not have, and the resulting plugin failed to load with
# "GLIBC_2.35 not found". Never take this baseline from the machine doing the building.
#
# Run this on every Linux artifact before it is handed to anyone.
#
#   scripts/check-glibc-baseline.sh <binary> [max glibc version]

set -euo pipefail

binary="${1:-}"
max_version="${2:-2.28}"

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

  # Name the symbols, not just the version. A bare version number leaves the reader with
  # no idea whether the cause is one stray call they can avoid or the C++ runtime itself,
  # and that is the entire difference between a five-minute fix and an afternoon. This is
  # an addition to the vendored warp-drive original and is worth porting back.
  echo >&2
  echo "Symbols above the baseline:" >&2
  objdump -T "$binary" 2>/dev/null |
    grep -o 'GLIBC_[0-9][0-9.]*[[:space:]].*' |
    awk '{print $1, $NF}' |
    sed 's/^GLIBC_//' |
    sort -uV |
    while read -r version symbol; do
      if version_gt "$version" "$max_version"; then
        printf '  %-10s %s\n' "$version" "$symbol" >&2
      fi
    done

  echo >&2
  echo "This binary will not load on a host whose glibc is older than $highest." >&2
  echo "Build it in a container matching the target -- see .github/workflows/ci.yml." >&2
  exit 1
fi

echo "OK: within the baseline."
