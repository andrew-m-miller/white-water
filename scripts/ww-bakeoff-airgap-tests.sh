#!/usr/bin/env bash
# Self-contained contract test for scripts/ww-bakeoff-airgap.

set -euo pipefail
IFS=$'\n\t'

SCRIPT_DIR=$(CDPATH= cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
WRAPPER="$SCRIPT_DIR/ww-bakeoff-airgap"
[[ -x "$WRAPPER" ]] || { printf 'wrapper is not executable: %s\n' "$WRAPPER" >&2; exit 1; }

die() { printf 'ww-bakeoff-airgap-tests: %s\n' "$*" >&2; exit 1; }
sha256_file() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | awk '{print $1}'; else shasum -a 256 "$1" | awk '{print $1}'; fi
}
count_lines() { [[ -f "$1" ]] && wc -l < "$1" | tr -d ' ' || printf '0\n'; }
count_python() {
  if [[ ! -f "$1" ]]; then
    printf '0\n'
    return
  fi
  awk '$0 == "python" { count++ } END { print count + 0 }' "$1"
}

TEMP_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/whitewater-airgap-test.XXXXXX")
TEMP_ROOT=$(CDPATH= cd -P -- "$TEMP_ROOT" && pwd -P)
trap 'rm -rf "$TEMP_ROOT"' EXIT

PACKAGE="$TEMP_ROOT/package"
mkdir -p "$PACKAGE/scripts" "$PACKAGE/tools/bakeoff" "$PACKAGE/runtime/source/bin"
cp "$WRAPPER" "$PACKAGE/scripts/ww-bakeoff-airgap"
chmod 0755 "$PACKAGE/scripts/ww-bakeoff-airgap"
printf '# fake evaluator entrypoint\n' > "$PACKAGE/tools/bakeoff/evaluator.py"
printf '# fake P25-6 profile driver entrypoint\n' > "$PACKAGE/tools/bakeoff/run.py"

UNPACK_LOG="$TEMP_ROOT/unpack.log"
PYTHON_LOG="$TEMP_ROOT/python.log"
NO_PYTHON_LOG="$TEMP_ROOT/no-python.log"
printf 'evaluator\n' > "$NO_PYTHON_LOG"
[[ "$(count_python "$NO_PYTHON_LOG")" == 0 ]] || die "zero-match count_python result was not exactly zero"
cat > "$PACKAGE/runtime/source/bin/conda-unpack" <<'FAKE_UNPACK'
#!/usr/bin/env python
# The fake bundled Python below handles this marker.  This shebang deliberately matches
# conda-pack so the wrapper test proves the relocated runtime is present on PATH.
FAKE_UNPACK
cat > "$PACKAGE/runtime/source/bin/python" <<'FAKE_PYTHON'
#!/usr/bin/env bash
set -euo pipefail
if [[ "${1-}" == */bin/conda-unpack ]]; then
  printf 'unpack\n' >> "$WW_TEST_UNPACK_LOG"
  exit 0
fi
printf 'python\n' >> "$WW_TEST_PYTHON_LOG"
printf 'PYTHON_ARGS=' >> "$WW_TEST_PYTHON_LOG"
for arg in "$@"; do printf '<%s>' "$arg" >> "$WW_TEST_PYTHON_LOG"; done
printf '\nPYTHONPATH=%s\nPIP_INDEX_URL=%s\n' "${PYTHONPATH-UNSET}" "${PIP_INDEX_URL-UNSET}" >> "$WW_TEST_PYTHON_LOG"
FAKE_PYTHON
chmod 0755 "$PACKAGE/runtime/source/bin/conda-unpack" "$PACKAGE/runtime/source/bin/python"
mkdir -p "$PACKAGE/runtime"
tar -cf "$PACKAGE/runtime/runtime.tar" -C "$PACKAGE/runtime/source" bin
rm -rf "$PACKAGE/runtime/source"

RUNTIME_ENV="$TEMP_ROOT/runtime-env"
run_wrapper() {
  WW_BAKEOFF_RUNTIME_ARCHIVE="$PACKAGE/runtime/runtime.tar" \
  WW_BAKEOFF_RUNTIME_ENV="$RUNTIME_ENV" \
  WW_TEST_UNPACK_LOG="$UNPACK_LOG" \
  WW_TEST_PYTHON_LOG="$PYTHON_LOG" \
  "$PACKAGE/scripts/ww-bakeoff-airgap" "$@"
}

run_wrapper verify --token 'hello world' --flag
[[ "$(count_lines "$UNPACK_LOG")" == 1 ]] || die "first invocation did not run conda-unpack exactly once"
[[ "$(count_python "$PYTHON_LOG")" == 1 ]] || die "first invocation did not exec bundled python"
PACKAGE_CANONICAL=$(CDPATH= cd -P -- "$PACKAGE" && pwd -P)
grep -F "PYTHON_ARGS=<$PACKAGE_CANONICAL/tools/bakeoff/evaluator.py><verify><--token><hello world><--flag>" "$PYTHON_LOG" >/dev/null || die "arguments were not forwarded exactly"
grep -F 'PYTHONPATH=UNSET' "$PYTHON_LOG" >/dev/null || die "PYTHONPATH was not sanitized"
grep -F 'PIP_INDEX_URL=UNSET' "$PYTHON_LOG" >/dev/null || die "network package index was not sanitized"
[[ "$(cat "$RUNTIME_ENV/.ww-bakeoff-unpack.sha256")" == "$(sha256_file "$PACKAGE/runtime/runtime.tar")" ]] || die "unpack marker is not bound to archive SHA"
[[ "$(stat -c '%a' "$RUNTIME_ENV/.ww-bakeoff-unpack.sha256" 2>/dev/null || stat -f '%Lp' "$RUNTIME_ENV/.ww-bakeoff-unpack.sha256")" == 644 ]] || die "unpack marker mode is not 0644"

run_wrapper verify --second-run
[[ "$(count_lines "$UNPACK_LOG")" == 1 ]] || die "matching archive reran conda-unpack"
[[ "$(count_python "$PYTHON_LOG")" == 2 ]] || die "second invocation did not exec bundled python"

# A relocated copy gets a new writable runtime root and must bootstrap independently.
RELOCATED="$TEMP_ROOT/relocated"
cp -R "$PACKAGE" "$RELOCATED"
RELOCATED_ENV="$TEMP_ROOT/relocated-runtime-env"
RELOCATED_UNPACK="$TEMP_ROOT/relocated-unpack.log"
RELOCATED_PYTHON="$TEMP_ROOT/relocated-python.log"
WW_BAKEOFF_RUNTIME_ARCHIVE="$RELOCATED/runtime/runtime.tar" \
WW_BAKEOFF_RUNTIME_ENV="$RELOCATED_ENV" \
WW_TEST_UNPACK_LOG="$RELOCATED_UNPACK" \
WW_TEST_PYTHON_LOG="$RELOCATED_PYTHON" \
"$RELOCATED/scripts/ww-bakeoff-airgap" verify --relocated
[[ "$(count_lines "$RELOCATED_UNPACK")" == 1 ]] || die "relocated package did not unpack"
RELOCATED_CANONICAL=$(CDPATH= cd -P -- "$RELOCATED" && pwd -P)
grep -F "PYTHON_ARGS=<$RELOCATED_CANONICAL/tools/bakeoff/evaluator.py><verify><--relocated>" "$RELOCATED_PYTHON" >/dev/null || die "relocated evaluator path was not forwarded"

# WW_BAKEOFF_ENTRYPOINT launches a different carried bake-off module (the P25-6 resumable
# profile driver) while the default (no env var set, exercised above) stays evaluator.py.
ENTRYPOINT_ENV="$TEMP_ROOT/entrypoint-runtime-env"
ENTRYPOINT_UNPACK_LOG="$TEMP_ROOT/entrypoint-unpack.log"
ENTRYPOINT_PYTHON_LOG="$TEMP_ROOT/entrypoint-python.log"
WW_BAKEOFF_RUNTIME_ARCHIVE="$PACKAGE/runtime/runtime.tar" \
WW_BAKEOFF_RUNTIME_ENV="$ENTRYPOINT_ENV" \
WW_BAKEOFF_ENTRYPOINT="tools/bakeoff/run.py" \
WW_TEST_UNPACK_LOG="$ENTRYPOINT_UNPACK_LOG" \
WW_TEST_PYTHON_LOG="$ENTRYPOINT_PYTHON_LOG" \
"$PACKAGE/scripts/ww-bakeoff-airgap" --selection selection.json --output-dir out
[[ "$(count_lines "$ENTRYPOINT_UNPACK_LOG")" == 1 ]] || die "driver entrypoint invocation did not unpack the runtime"
grep -F "PYTHON_ARGS=<-m><tools.bakeoff.run><--selection><selection.json><--output-dir><out>" "$ENTRYPOINT_PYTHON_LOG" >/dev/null \
  || die "WW_BAKEOFF_ENTRYPOINT did not launch the driver as -m tools.bakeoff.run with forwarded arguments"
grep -F "PYTHONPATH=$PACKAGE_CANONICAL" "$ENTRYPOINT_PYTHON_LOG" >/dev/null \
  || die "driver module invocation did not set PYTHONPATH to the package root"

# An unrecognized, absolute, or symlinked entrypoint must be rejected before Python ever runs.
if WW_BAKEOFF_RUNTIME_ARCHIVE="$PACKAGE/runtime/runtime.tar" WW_BAKEOFF_RUNTIME_ENV="$TEMP_ROOT/entrypoint-bad-env" \
   WW_BAKEOFF_ENTRYPOINT="tools/bakeoff/not-a-real-module.py" \
   WW_TEST_UNPACK_LOG="$UNPACK_LOG" WW_TEST_PYTHON_LOG="$PYTHON_LOG" \
   "$PACKAGE/scripts/ww-bakeoff-airgap" verify --unsupported-entrypoint; then
  die "an unrecognized WW_BAKEOFF_ENTRYPOINT was accepted"
fi
if WW_BAKEOFF_RUNTIME_ARCHIVE="$PACKAGE/runtime/runtime.tar" WW_BAKEOFF_RUNTIME_ENV="$TEMP_ROOT/entrypoint-abs-env" \
   WW_BAKEOFF_ENTRYPOINT="/etc/passwd" \
   WW_TEST_UNPACK_LOG="$UNPACK_LOG" WW_TEST_PYTHON_LOG="$PYTHON_LOG" \
   "$PACKAGE/scripts/ww-bakeoff-airgap" verify --absolute-entrypoint; then
  die "an absolute WW_BAKEOFF_ENTRYPOINT was accepted"
fi
# An allowlisted entrypoint name that resolves to a symlink (not a real carried module) must
# still be rejected -- the allowlist alone is not sufficient if the file itself was replaced.
SYMLINK_PACKAGE="$TEMP_ROOT/symlink-package"
cp -R "$PACKAGE" "$SYMLINK_PACKAGE"
rm -f "$SYMLINK_PACKAGE/tools/bakeoff/run.py"
ln -s evaluator.py "$SYMLINK_PACKAGE/tools/bakeoff/run.py"
if WW_BAKEOFF_RUNTIME_ARCHIVE="$SYMLINK_PACKAGE/runtime/runtime.tar" WW_BAKEOFF_RUNTIME_ENV="$TEMP_ROOT/entrypoint-symlink-env" \
   WW_BAKEOFF_ENTRYPOINT="tools/bakeoff/run.py" \
   WW_TEST_UNPACK_LOG="$UNPACK_LOG" WW_TEST_PYTHON_LOG="$PYTHON_LOG" \
   "$SYMLINK_PACKAGE/scripts/ww-bakeoff-airgap" verify --symlink-entrypoint; then
  die "a symlinked WW_BAKEOFF_ENTRYPOINT target was accepted"
fi

# Refuse a package-root extraction target and a non-empty unmarked target.
if WW_BAKEOFF_RUNTIME_ARCHIVE="$PACKAGE/runtime/runtime.tar" WW_BAKEOFF_RUNTIME_ENV="$PACKAGE" \
   WW_TEST_UNPACK_LOG="$UNPACK_LOG" WW_TEST_PYTHON_LOG="$PYTHON_LOG" \
   "$PACKAGE/scripts/ww-bakeoff-airgap" verify --unsafe-root; then
  die "package-root runtime target was accepted"
fi
NONEMPTY="$TEMP_ROOT/nonempty-runtime"
mkdir -p "$NONEMPTY"
printf 'keep\n' > "$NONEMPTY/keep.txt"
if WW_BAKEOFF_RUNTIME_ARCHIVE="$PACKAGE/runtime/runtime.tar" WW_BAKEOFF_RUNTIME_ENV="$NONEMPTY" \
   WW_TEST_UNPACK_LOG="$UNPACK_LOG" WW_TEST_PYTHON_LOG="$PYTHON_LOG" \
   "$PACKAGE/scripts/ww-bakeoff-airgap" verify --nonempty; then
  die "non-empty unmarked runtime target was overwritten"
fi
[[ -f "$NONEMPTY/keep.txt" ]] || die "non-empty runtime target was modified"

# Reject archive traversal before extraction.
BAD_SOURCE="$TEMP_ROOT/bad-source"
mkdir -p "$BAD_SOURCE"
printf 'escape\n' > "$TEMP_ROOT/escape.txt"
tar -cf "$TEMP_ROOT/bad.tar" -C "$BAD_SOURCE" ../escape.txt
BAD_ENV="$TEMP_ROOT/bad-env"
if WW_BAKEOFF_RUNTIME_ARCHIVE="$TEMP_ROOT/bad.tar" WW_BAKEOFF_RUNTIME_ENV="$BAD_ENV" \
   WW_TEST_UNPACK_LOG="$UNPACK_LOG" WW_TEST_PYTHON_LOG="$PYTHON_LOG" \
   "$PACKAGE/scripts/ww-bakeoff-airgap" verify --bad-archive; then
  die "archive traversal member was accepted"
fi
[[ ! -e "$BAD_ENV/bin/python" ]] || die "unsafe archive was extracted"

# A changed archive cannot silently reuse or overwrite an existing runtime-env.
printf 'changed\n' >> "$PACKAGE/runtime/runtime.tar"
if run_wrapper verify --changed-archive; then
  die "changed archive reused an old runtime-env"
fi
[[ "$(count_lines "$UNPACK_LOG")" == 1 ]] || die "changed archive reran conda-unpack"

printf 'ww-bakeoff-airgap-tests: PASS\n'
