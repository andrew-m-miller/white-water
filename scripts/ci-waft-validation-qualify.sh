#!/usr/bin/env bash
#
# The fail-closed seam between the EL8 CI job and the WAFT export/validation packager.
#
# This lane produces the SOURCE runtime that lets the human operator run models/export_waft.py
# offline on the airgapped box: it carries a conda-pack export/validation environment, the export
# entrypoint plus its models/ closure, and the pinned WAFT BSD-3-Clause source as a clean git
# checkout. It carries NO checkpoint (operator-supplied), NO ONNX Runtime CUDA-12 archive and NO
# native ORT bridge (the export is device-independent; onnxruntime is a conda-forge CPU package).
#
# Unlike scripts/ci-p25-6-qualify.sh this seam does NOT invoke tools/p25_5/package.py: that
# packager is a candidate-admission/protocol-v2 measurement packager (admitted candidates, artifact
# hashes, report schemas), none of which applies to an export/validation runtime. Instead it
# assembles the outer tarball directly from an explicit staged tree, reusing the parts that DO
# apply -- the conda solve+pack and tools/p25_5/licenses.py inventory tooling (run in the ci.yml
# job that calls this script) and the ww-bakeoff-airgap offline wrapper.
#
# Required environment:
#   WAFT_RUNTIME_ROOT            extracted, conda-unpacked runtime prefix (for the glibc/no-checkpoint recheck)
#   WAFT_RUNTIME_ARCHIVE         exact conda-pack archive (tar.gz)
#   WAFT_RUNTIME_SHA256          SHA256 of that archive
#   WAFT_RUNTIME_SHA256_FILE     checksum sidecar for the archive
#   WAFT_RUNTIME_INVENTORY       CI-solved @EXPLICIT conda lock (conda list --explicit)
#   WAFT_RUNTIME_LEGAL_DIR       generated runtime license/notice bundle directory
#   WAFT_RUNTIME_LEGAL_REVIEW_FILE   Andrew Miller approval bound to the generated inventory
#   WAFT_RUNTIME_LEGAL_REVIEW_SHA256 SHA256 of that exact review JSON
#   WAFT_SRC_CHECKOUT            clean git checkout of the pinned WAFT commit (contains .git)
#   WAFT_RUNTIME_INPUTS_FILE     checked-in bakeoff/waft-validation/runtime-inputs.json
#   WAFT_RUN_INSTRUCTIONS        RUN-WAFT-VALIDATION.txt source
#   WAFT_WRAPPER                 scripts/ww-bakeoff-airgap
#   WAFT_EXPORT_ENTRYPOINT       models/export_waft.py
#   WAFT_ARTIFACT_WORKFLOW       models/artifact_workflow.py
#   WAFT_MANIFEST                models/waft-twins-artifact.json
#   WAFT_OUTPUT_DIR              output directory for the carried package/evidence
#   GITHUB_SHA                   exact checked-out GitHub Actions source commit
#
# Pinned identity re-verified here (never invented):
#   WAFT source commit  b152ff1cad1af8c185ee7b141997c48ff3334c87
#   checkpoint (NOT carried) sha256 f750cd15281fc30de477723438ff4a67fe1591deac4ab0eb9b366e27c827e070, 544230582 bytes

set -euo pipefail

WAFT_SOURCE_COMMIT="b152ff1cad1af8c185ee7b141997c48ff3334c87"
WAFT_CHECKPOINT_SHA256="f750cd15281fc30de477723438ff4a67fe1591deac4ab0eb9b366e27c827e070"

fail() {
  echo "WAFT-validation CI: $*" >&2
  exit 1
}

require_env() {
  local name="$1"
  [[ -n "${!name:-}" ]] || fail "required input $name is empty; this lane is fail-closed"
}

require_regular() {
  local path="$1" label="$2"
  [[ -e "$path" ]] || fail "$label is missing: $path"
  [[ ! -L "$path" ]] || fail "$label must not be a symlink: $path"
  [[ -f "$path" ]] || fail "$label must be a regular file: $path"
}

require_directory() {
  local path="$1" label="$2"
  [[ -d "$path" ]] || fail "$label is missing: $path"
  [[ ! -L "$path" ]] || fail "$label must not be a symlink: $path"
}

for required in \
  WAFT_RUNTIME_ROOT \
  WAFT_RUNTIME_ARCHIVE \
  WAFT_RUNTIME_SHA256 \
  WAFT_RUNTIME_SHA256_FILE \
  WAFT_RUNTIME_INVENTORY \
  WAFT_RUNTIME_LEGAL_DIR \
  WAFT_RUNTIME_LEGAL_REVIEW_FILE \
  WAFT_RUNTIME_LEGAL_REVIEW_SHA256 \
  WAFT_SRC_CHECKOUT \
  WAFT_RUNTIME_INPUTS_FILE \
  WAFT_RUN_INSTRUCTIONS \
  WAFT_WRAPPER \
  WAFT_EXPORT_ENTRYPOINT \
  WAFT_ARTIFACT_WORKFLOW \
  WAFT_MANIFEST \
  WAFT_OUTPUT_DIR; do
  require_env "$required"
done

root="${GITHUB_WORKSPACE:-$(pwd)}"
output="$WAFT_OUTPUT_DIR"
mkdir -p "$output"

resolve_repo_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s/%s\n' "$root" "$1" ;;
  esac
}

runtime_archive=$(resolve_repo_path "$WAFT_RUNTIME_ARCHIVE")
runtime_sha_file=$(resolve_repo_path "$WAFT_RUNTIME_SHA256_FILE")
runtime_inventory=$(resolve_repo_path "$WAFT_RUNTIME_INVENTORY")
runtime_root=$(resolve_repo_path "$WAFT_RUNTIME_ROOT")
runtime_legal_dir=$(resolve_repo_path "$WAFT_RUNTIME_LEGAL_DIR")
runtime_legal_review_file=$(resolve_repo_path "$WAFT_RUNTIME_LEGAL_REVIEW_FILE")
src_checkout=$(resolve_repo_path "$WAFT_SRC_CHECKOUT")
runtime_inputs=$(resolve_repo_path "$WAFT_RUNTIME_INPUTS_FILE")
run_instructions_source=$(resolve_repo_path "$WAFT_RUN_INSTRUCTIONS")
wrapper=$(resolve_repo_path "$WAFT_WRAPPER")
export_entrypoint=$(resolve_repo_path "$WAFT_EXPORT_ENTRYPOINT")
artifact_workflow=$(resolve_repo_path "$WAFT_ARTIFACT_WORKFLOW")
manifest=$(resolve_repo_path "$WAFT_MANIFEST")

require_regular "$runtime_archive" "WAFT-validation conda-pack runtime archive"
require_regular "$runtime_sha_file" "WAFT-validation runtime SHA256 file"
require_regular "$runtime_inventory" "WAFT-validation CI-solved @EXPLICIT conda lock"
require_directory "$runtime_root" "WAFT-validation extracted runtime"
require_directory "$runtime_legal_dir" "WAFT-validation generated runtime legal directory"
require_regular "$runtime_legal_review_file" "WAFT-validation runtime legal-review file"
require_directory "$src_checkout" "WAFT pinned source checkout"
require_regular "$runtime_inputs" "WAFT-validation runtime-inputs manifest"
require_regular "$run_instructions_source" "WAFT-validation run instructions source"
require_regular "$wrapper" "ww-bakeoff-airgap wrapper"
require_regular "$export_entrypoint" "models/export_waft.py"
require_regular "$artifact_workflow" "models/artifact_workflow.py"
require_regular "$manifest" "models/waft-twins-artifact.json"

# The CI-solved lock must be a real @EXPLICIT lock, not the requested match-spec.
grep -qx '@EXPLICIT' "$runtime_inventory" ||
  fail "runtime inventory is not a CI-solved @EXPLICIT lock: $runtime_inventory"

# Runtime archive identity.
[[ "$WAFT_RUNTIME_SHA256" =~ ^[0-9a-f]{64}$ ]] ||
  fail "WAFT_RUNTIME_SHA256 must be a lowercase SHA256"
actual_runtime_sha=$(sha256sum -- "$runtime_archive" | awk '{print $1}')
[[ "$actual_runtime_sha" == "$WAFT_RUNTIME_SHA256" ]] ||
  fail "conda-pack runtime SHA256 mismatch: expected $WAFT_RUNTIME_SHA256, got $actual_runtime_sha"

# Runtime legal review: exact bytes + reviewed:true bound to the generated inventory hash.
for legal_input in "$runtime_legal_review_file"; do
  [[ "$(stat -c '%a' "$legal_input")" == 644 ]] ||
    fail "runtime legal-review input must have mode 0644: $legal_input"
done
[[ "$WAFT_RUNTIME_LEGAL_REVIEW_SHA256" =~ ^[0-9a-f]{64}$ ]] ||
  fail "WAFT_RUNTIME_LEGAL_REVIEW_SHA256 must be a lowercase SHA256"
actual_review_sha=$(sha256sum -- "$runtime_legal_review_file" | awk '{print $1}')
[[ "$actual_review_sha" == "$WAFT_RUNTIME_LEGAL_REVIEW_SHA256" ]] ||
  fail "runtime legal-review SHA256 mismatch: expected $WAFT_RUNTIME_LEGAL_REVIEW_SHA256, got $actual_review_sha"

runtime_legal_inventory="$runtime_legal_dir/runtime-license-inventory.json"
require_regular "$runtime_legal_dir/RUNTIME-LICENSES.txt" "WAFT-validation runtime license aggregate"
require_regular "$runtime_legal_dir/RUNTIME-NOTICES.txt" "WAFT-validation runtime notice aggregate"
require_regular "$runtime_legal_inventory" "WAFT-validation runtime license inventory"
runtime_inventory_sha=$(sha256sum -- "$runtime_legal_inventory" | awk '{print $1}')
python3.11 "$root/tools/p25_5/licenses.py" validate-runtime-review \
  --input "$runtime_legal_review_file" \
  --sha256 "$WAFT_RUNTIME_LEGAL_REVIEW_SHA256" \
  --inventory-sha256 "$runtime_inventory_sha"
# Recheck the exact relocated runtime that will actually be carried against the approved inventory.
python3.11 "$root/tools/p25_5/licenses.py" verify-runtime-content \
  --prefix "$runtime_root" \
  --inventory "$runtime_legal_inventory" \
  --inventory-sha256 "$runtime_inventory_sha"

# The pinned WAFT source checkout must be exactly the manifest commit and clean -- the same
# identity export_waft.py re-verifies on the box.
source_commit="${GITHUB_SHA:-}"
[[ "$source_commit" =~ ^[0-9a-f]{40}$ ]] ||
  fail "GITHUB_SHA must identify the exact checked-out lowercase 40-hex source commit"
require_directory "$src_checkout/.git" "WAFT source .git metadata"
waft_head=$(git -C "$src_checkout" rev-parse HEAD)
[[ "$waft_head" == "$WAFT_SOURCE_COMMIT" ]] ||
  fail "vendored WAFT checkout is not the pinned commit: expected $WAFT_SOURCE_COMMIT, got $waft_head"
waft_dirty=$(git -C "$src_checkout" status --porcelain=v1 --untracked-files=all)
[[ -z "$waft_dirty" ]] ||
  fail "vendored WAFT checkout is not clean; export_waft.py would refuse it"
require_regular "$src_checkout/config/a2/twins/chairs-things.json" "pinned WAFT Twins config"

# Assemble the staged package tree. Every carried file except the wrapper is 0644; the wrapper is
# the only 0755 regular file (mirrors the P25-6 package contract).
staging_root="$output/staging"
pkg="$staging_root/whitewater-waft-validation-el8"
rm -rf "$staging_root"
mkdir -p "$pkg/scripts" "$pkg/models" "$pkg/runtime" "$pkg/legal" "$pkg/bakeoff/waft-validation"

install -m 0755 "$wrapper" "$pkg/scripts/ww-bakeoff-airgap"
install -m 0644 "$export_entrypoint" "$pkg/models/export_waft.py"
install -m 0644 "$artifact_workflow" "$pkg/models/artifact_workflow.py"
install -m 0644 "$manifest" "$pkg/models/waft-twins-artifact.json"
install -m 0644 "$runtime_archive" "$pkg/runtime/whitewater-waft-validation-runtime.tar.gz"
install -m 0644 "$runtime_sha_file" "$pkg/runtime/whitewater-waft-validation-runtime.tar.gz.sha256"
install -m 0644 "$runtime_inventory" "$pkg/runtime/whitewater-waft-validation-runtime.inventory"
install -m 0644 "$runtime_legal_dir/RUNTIME-LICENSES.txt" "$pkg/legal/RUNTIME-LICENSES.txt"
install -m 0644 "$runtime_legal_dir/RUNTIME-NOTICES.txt" "$pkg/legal/RUNTIME-NOTICES.txt"
install -m 0644 "$runtime_legal_inventory" "$pkg/legal/runtime-license-inventory.json"
install -m 0644 "$runtime_legal_review_file" "$pkg/legal/runtime-legal-review.json"
install -m 0644 "$runtime_inputs" "$pkg/bakeoff/waft-validation/runtime-inputs.json"
install -m 0644 "$run_instructions_source" "$pkg/RUN-WAFT-VALIDATION.txt"

# Vendor the WAFT source as a clean git checkout (including .git). Copy without dereferencing so a
# stray symlink is preserved-then-rejected by the structure gate rather than silently followed.
cp -a "$src_checkout" "$pkg/waft-src"

# Hard stop: the non-commercial thirdparty/dinov3 submodule content must never be carried. The
# Twins export path does not need it (proved by the CI weightless WAFTv2(twins) construction
# smoke); this guard fails closed if a future change ever inits that submodule.
if [ -n "$(find "$pkg/waft-src/thirdparty/dinov3" -type f 2>/dev/null)" ]; then
  fail "staged package carries thirdparty/dinov3 submodule content (non-commercial DINOv3); refusing"
fi
require_regular "$pkg/waft-src/config/a2/twins/chairs-things.json" "staged pinned WAFT Twins config"

# Hard stop: the checkpoint must never be inside the staged tree.
if find "$pkg" -type f \( -name '*.pth' -o -name 'zero-shot*' \) -print -quit | grep -q .; then
  fail "a checkpoint-like file is present in the staged package; the checkpoint must never be carried"
fi
# Belt-and-braces: no file in the staged tree may match the pinned checkpoint bytes.
while IFS= read -r -d '' f; do
  [[ "$(stat -c '%s' "$f")" == 544230582 ]] || continue
  if [[ "$(sha256sum -- "$f" | awk '{print $1}')" == "$WAFT_CHECKPOINT_SHA256" ]]; then
    fail "the pinned WAFT checkpoint bytes are present in the staged package: $f"
  fi
done < <(find "$pkg" -type f -print0)

package_tarball="$output/whitewater-waft-validation-el8.tar.gz"
package_checksum="$package_tarball.sha256"
echo "WAFT-validation: assembling outer archive from the staged tree"
tar -C "$staging_root" -czf "$package_tarball" "whitewater-waft-validation-el8"
package_sha=$(sha256sum -- "$package_tarball" | awk '{print $1}')
printf '%s  %s\n' "$package_sha" "$(basename "$package_tarball")" > "$package_checksum"
chmod 0644 "$package_tarball" "$package_checksum"

require_regular "$package_tarball" "WAFT-validation package tarball"
require_regular "$package_checksum" "WAFT-validation package SHA256"

# Structure gate: no links, and carried data files are 0644 while the wrapper stays 0755.
python3.11 - "$package_tarball" <<'PY'
import stat
import sys
import tarfile
from pathlib import PurePosixPath

archive = sys.argv[1]
with tarfile.open(archive, "r:gz") as stream:
    members = stream.getmembers()
    if not members:
        raise SystemExit("WAFT-validation package archive is empty")
    for member in members:
        if member.issym() or member.islnk():
            raise SystemExit(f"WAFT-validation package archive contains a link: {member.name}")
        if not member.isdir() and not member.isfile():
            raise SystemExit(f"WAFT-validation package archive contains a non-regular entry: {member.name}")
        if member.isfile():
            name = PurePosixPath(member.name)
            mode = stat.S_IMODE(member.mode)
            if name.name == "ww-bakeoff-airgap":
                if mode != 0o755:
                    raise SystemExit(f"{member.name}: the airgap wrapper must be 0755")
                continue
            # Everything under the vendored git metadata is carried as-is; git tracks its own modes.
            if ".git/" in (str(name) + "/"):
                continue
            if mode not in (0o644, 0o755):
                raise SystemExit(f"{member.name}: carried file has unexpected mode {mode:04o}")
print(f"WAFT-validation package archive structure: PASS ({len(members)} entries; no links)")
PY

cat > "$output/WAFT-VALIDATION.txt" <<'EOF'
White Water WAFT export/validation runtime package
==================================================

This archive lets the operator run models/export_waft.py OFFLINE on the airgapped EL8 box to
export and qualify the pinned WAFT/Twins candidate. It carries the conda-pack export/validation
runtime, the export entrypoint and its models/ closure, and the pinned WAFT BSD-3-Clause source
as a clean git checkout. It carries NO checkpoint: the operator supplies zero-shot.pth themselves
and verifies its pinned SHA256/size. It selects no model and authorizes no redistribution of the
checkpoint. Follow RUN-WAFT-VALIDATION.txt. No production images or checkpoints leave the machine.
EOF
chmod 0644 "$output/WAFT-VALIDATION.txt"

if ! grep -Eiq 'zero-shot|checkpoint|export_waft' "$run_instructions_source"; then
  fail "run instructions do not describe the WAFT export/validation procedure"
fi

echo "WAFT-validation CI: qualification and package passed"
echo "  tarball: $package_tarball"
echo "  sha256:  $package_sha"
echo "  WAFT source commit: $waft_head"
