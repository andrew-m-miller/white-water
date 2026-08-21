#!/usr/bin/env bash
# Report the exact ELF dependency closure seen by the loader for an ORT CUDA payload.
# Run this on the EL8 artifact build or target Flame box, not on macOS.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/report-ort-closure.sh <payload-directory>

Reports every ELF file and direct DT_NEEDED entry in the payload, the loader-resolved
transitive closure, unresolved libraries, unique resolved bytes outside the payload, and
the payload's on-disk size. Run it in the same Linux environment used to load the probe;
ldd resolution is environment-dependent.
EOF
}

if [[ ${1:-} == "--help" || ${1:-} == "-h" ]]; then
  usage
  exit 0
fi
if [[ $# -ne 1 ]]; then
  usage >&2
  exit 2
fi
if [[ $(uname -s) != "Linux" ]]; then
  echo "error: dependency closure must be measured on Linux (current: $(uname -s))" >&2
  exit 2
fi

for tool in awk du find ldd readelf readlink sha256sum sort stat; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "error: required tool not found: $tool" >&2
    exit 2
  fi
done

payload=$(readlink -f -- "$1")
if [[ ! -d $payload ]]; then
  echo "error: payload directory does not exist: $1" >&2
  exit 2
fi

work=$(mktemp -d)
trap 'rm -rf -- "$work"' EXIT
elfs="$work/elfs"
resolved="$work/resolved"
missing="$work/missing"
: >"$elfs"
: >"$resolved"
: >"$missing"

while IFS= read -r -d '' candidate; do
  if readelf -h -- "$candidate" >/dev/null 2>&1; then
    printf '%s\n' "$candidate" >>"$elfs"
  fi
done < <(find "$payload" -type f -print0)

if [[ ! -s $elfs ]]; then
  echo "error: no ELF files found under: $payload" >&2
  exit 1
fi

# ORT providers locate siblings beside the core runtime. Include every directory containing
# an ELF payload file so ldd resolves the bundle as it will be staged, while preserving any
# caller-supplied search path after it.
search_path=$(while IFS= read -r elf; do dirname "$elf"; done <"$elfs" | sort -u |
  awk 'BEGIN { ORS="" } NR > 1 { printf ":" } { printf "%s", $0 }')
if [[ -n ${LD_LIBRARY_PATH:-} ]]; then
  search_path+="${search_path:+:}${LD_LIBRARY_PATH}"
fi

echo "ORT CUDA payload closure"
echo "payload: $payload"
echo "loader search path: $search_path"
echo "payload apparent bytes: $(du -sb -- "$payload" | awk '{print $1}')"
echo "payload allocated bytes: $(du -s -B1 -- "$payload" | awk '{print $1}')"
echo
echo "ELF payload files and direct DT_NEEDED entries"

while IFS= read -r elf; do
  relative=${elf#"$payload"/}
  echo
  echo "$relative"
  echo "  bytes: $(stat -c %s -- "$elf")"
  echo "  sha256: $(sha256sum -- "$elf" | awk '{print $1}')"
  needed=$(readelf -d -- "$elf" | awk '
    /\(NEEDED\)/ {
      value=$0
      sub(/^.*\[/, "", value)
      sub(/\].*$/, "", value)
      print value
    }')
  if [[ -n $needed ]]; then
    while IFS= read -r name; do echo "  NEEDED: $name"; done <<<"$needed"
  else
    echo "  NEEDED: <none>"
  fi

  while IFS= read -r line; do
    if [[ $line =~ ^[[:space:]]*([^[:space:]]+)[[:space:]]+\=\>[[:space:]]+not[[:space:]]+found ]]; then
      printf '%s\t%s\n' "$relative" "${BASH_REMATCH[1]}" >>"$missing"
    elif [[ $line =~ \=\>[[:space:]]+(/[^[:space:]]+) ]]; then
      printf '%s\n' "${BASH_REMATCH[1]}" >>"$resolved"
    elif [[ $line =~ ^[[:space:]]*(/[^[:space:]]+) ]]; then
      printf '%s\n' "${BASH_REMATCH[1]}" >>"$resolved"
    fi
  done < <(LD_LIBRARY_PATH="$search_path" ldd -- "$elf" 2>&1 || true)
done <"$elfs"

canonical_resolved="$work/resolved-canonical"
while IFS= read -r library; do
  readlink -f -- "$library"
done <"$resolved" | sort -u >"$canonical_resolved"
mv "$canonical_resolved" "$resolved"
sort -u -o "$missing" "$missing"

echo
echo "Unique loader-resolved transitive closure"
resolved_bytes=0
if [[ -s $resolved ]]; then
  while IFS= read -r library; do
    canonical=$(readlink -f -- "$library")
    bytes=$(stat -c %s -- "$canonical")
    resolved_bytes=$((resolved_bytes + bytes))
    if [[ $canonical == "$payload"/* ]]; then
      location="payload"
    else
      location="external"
    fi
    echo "  $location $bytes $canonical"
  done <"$resolved"
else
  echo "  <none>"
fi
echo "unique resolved closure bytes: $resolved_bytes"

echo
echo "Unresolved dependencies"
if [[ -s $missing ]]; then
  while IFS=$'\t' read -r owner name; do
    echo "  $owner -> $name"
  done <"$missing"
  exit 1
fi
echo "  <none>"
