#!/usr/bin/env bash
#
# The fail-closed seam between the EL8 CI job and the NeuFlow v2 export/validation packager.
#
# This lane produces the SOURCE runtime that lets the human operator run models/export_neuflow_v2.py
# offline on the airgapped box: it carries a conda-pack export/validation environment and the export
# entrypoint plus its models/ closure. It carries NO checkpoint (operator-supplied), NO ONNX Runtime
# CUDA-12 archive and NO native ORT bridge (the export is device-independent; the ML stack --
# torch/onnxruntime/etc. -- is pip-installed CPU wheels at CI build, not conda-sourced; the conda
# spec is a minimal base).
#
# It also -- unlike scripts/ci-waft-validation-qualify.sh -- carries NO vendored upstream source.
# NeuFlow v2 tracks its checkpoints (neuflow_mixed.pth and siblings, 36MB real git blobs) IN the
# upstream repo, so a clean vendored checkout would redistribute the unknown-licence checkpoint,
# while stripping the .pth files would dirty the worktree export_neuflow_v2.py requires clean. The
# licence-safe choice is to carry no upstream tree; the operator supplies their own clean NeuFlow_v2
# checkout (which inherently contains the checkpoint) on media. CI clones NeuFlow_v2 only to run the
# weightless construction smoke (in the ci.yml job that calls this script) and never packages it.
#
# Like scripts/ci-waft-validation-qualify.sh this seam does NOT invoke tools/p25_5/package.py (a
# candidate-admission/protocol-v2 measurement packager that does not apply to an export/validation
# runtime). It assembles the outer tarball directly from an explicit staged tree, reusing the parts
# that DO apply -- the conda solve+pack and tools/p25_5/licenses.py inventory tooling (run in the
# ci.yml job that calls this script) and the ww-bakeoff-airgap offline wrapper.
#
# Required environment:
#   NEUFLOW_RUNTIME_ROOT            extracted, conda-unpacked runtime prefix (for the glibc/no-checkpoint recheck)
#   NEUFLOW_RUNTIME_ARCHIVE         exact conda-pack archive (tar.gz)
#   NEUFLOW_RUNTIME_SHA256          SHA256 of that archive
#   NEUFLOW_RUNTIME_SHA256_FILE     checksum sidecar for the archive
#   NEUFLOW_RUNTIME_INVENTORY       CI-solved @EXPLICIT conda lock (conda list --explicit)
#   NEUFLOW_RUNTIME_LEGAL_DIR       generated runtime license/notice bundle directory
#   NEUFLOW_RUNTIME_LEGAL_REVIEW_FILE   Andrew Miller approval bound to the generated inventory
#   NEUFLOW_RUNTIME_LEGAL_REVIEW_SHA256 SHA256 of that exact review JSON
#   NEUFLOW_RUNTIME_INPUTS_FILE     checked-in bakeoff/neuflow-validation/runtime-inputs.json
#   NEUFLOW_RUN_INSTRUCTIONS        RUN-NEUFLOW-VALIDATION.txt source
#   NEUFLOW_WRAPPER                 scripts/ww-bakeoff-airgap
#   NEUFLOW_EXPORT_ENTRYPOINT       models/export_neuflow_v2.py
#   NEUFLOW_ARTIFACT_WORKFLOW       models/artifact_workflow.py
#   NEUFLOW_EXCLUSION_CONTRACT      models/exclusion_contract.py
#   NEUFLOW_MANIFEST               models/neuflow-v2.json
#   NEUFLOW_OUTPUT_DIR              output directory for the carried package/evidence
#
# Pinned identity re-verified here (never invented):
#   NeuFlow source commit  204b5e3744461d90303b9ff82caa7a1bb56a2ca2 (NOT carried; asserted absent)
#   checkpoint (NOT carried) sha256 76152c8068f247a7d073aa13e61da8cb4c3c6a798076d4dc8e20f7995fcc019f, 36195519 bytes

set -euo pipefail

NEUFLOW_SOURCE_COMMIT="204b5e3744461d90303b9ff82caa7a1bb56a2ca2"
NEUFLOW_CHECKPOINT_SHA256="76152c8068f247a7d073aa13e61da8cb4c3c6a798076d4dc8e20f7995fcc019f"
NEUFLOW_CHECKPOINT_SIZE="36195519"

fail() {
  echo "NeuFlow-validation CI: $*" >&2
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
  NEUFLOW_RUNTIME_ROOT \
  NEUFLOW_RUNTIME_ARCHIVE \
  NEUFLOW_RUNTIME_SHA256 \
  NEUFLOW_RUNTIME_SHA256_FILE \
  NEUFLOW_RUNTIME_INVENTORY \
  NEUFLOW_RUNTIME_LEGAL_DIR \
  NEUFLOW_RUNTIME_LEGAL_REVIEW_FILE \
  NEUFLOW_RUNTIME_LEGAL_REVIEW_SHA256 \
  NEUFLOW_RUNTIME_INPUTS_FILE \
  NEUFLOW_RUN_INSTRUCTIONS \
  NEUFLOW_WRAPPER \
  NEUFLOW_EXPORT_ENTRYPOINT \
  NEUFLOW_ARTIFACT_WORKFLOW \
  NEUFLOW_EXCLUSION_CONTRACT \
  NEUFLOW_MANIFEST \
  NEUFLOW_OUTPUT_DIR; do
  require_env "$required"
done

root="${GITHUB_WORKSPACE:-$(pwd)}"
output="$NEUFLOW_OUTPUT_DIR"
mkdir -p "$output"

resolve_repo_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s/%s\n' "$root" "$1" ;;
  esac
}

runtime_archive=$(resolve_repo_path "$NEUFLOW_RUNTIME_ARCHIVE")
runtime_sha_file=$(resolve_repo_path "$NEUFLOW_RUNTIME_SHA256_FILE")
runtime_inventory=$(resolve_repo_path "$NEUFLOW_RUNTIME_INVENTORY")
runtime_root=$(resolve_repo_path "$NEUFLOW_RUNTIME_ROOT")
runtime_legal_dir=$(resolve_repo_path "$NEUFLOW_RUNTIME_LEGAL_DIR")
runtime_legal_review_file=$(resolve_repo_path "$NEUFLOW_RUNTIME_LEGAL_REVIEW_FILE")
runtime_inputs=$(resolve_repo_path "$NEUFLOW_RUNTIME_INPUTS_FILE")
run_instructions_source=$(resolve_repo_path "$NEUFLOW_RUN_INSTRUCTIONS")
wrapper=$(resolve_repo_path "$NEUFLOW_WRAPPER")
export_entrypoint=$(resolve_repo_path "$NEUFLOW_EXPORT_ENTRYPOINT")
artifact_workflow=$(resolve_repo_path "$NEUFLOW_ARTIFACT_WORKFLOW")
exclusion_contract=$(resolve_repo_path "$NEUFLOW_EXCLUSION_CONTRACT")
manifest=$(resolve_repo_path "$NEUFLOW_MANIFEST")

require_regular "$runtime_archive" "NeuFlow-validation conda-pack runtime archive"
require_regular "$runtime_sha_file" "NeuFlow-validation runtime SHA256 file"
require_regular "$runtime_inventory" "NeuFlow-validation CI-solved @EXPLICIT conda lock"
require_directory "$runtime_root" "NeuFlow-validation extracted runtime"
require_directory "$runtime_legal_dir" "NeuFlow-validation generated runtime legal directory"
require_regular "$runtime_legal_review_file" "NeuFlow-validation runtime legal-review file"
require_regular "$runtime_inputs" "NeuFlow-validation runtime-inputs manifest"
require_regular "$run_instructions_source" "NeuFlow-validation run instructions source"
require_regular "$wrapper" "ww-bakeoff-airgap wrapper"
require_regular "$export_entrypoint" "models/export_neuflow_v2.py"
require_regular "$artifact_workflow" "models/artifact_workflow.py"
require_regular "$exclusion_contract" "models/exclusion_contract.py"
require_regular "$manifest" "models/neuflow-v2.json"

# The CI-solved lock must be a real @EXPLICIT lock, not the requested match-spec.
grep -qx '@EXPLICIT' "$runtime_inventory" ||
  fail "runtime inventory is not a CI-solved @EXPLICIT lock: $runtime_inventory"

# The carried candidate manifest must pin exactly the audited NeuFlow source commit -- the same
# commit the operator's --upstream checkout must be at (never invented here).
manifest_commit=$(python3.11 - "$manifest" <<'PY'
import json, sys
print(json.loads(open(sys.argv[1], encoding="utf-8").read())["upstream"]["commit"])
PY
)
[[ "$manifest_commit" == "$NEUFLOW_SOURCE_COMMIT" ]] ||
  fail "models/neuflow-v2.json pins a different NeuFlow commit: expected $NEUFLOW_SOURCE_COMMIT, got $manifest_commit"

# Runtime archive identity.
[[ "$NEUFLOW_RUNTIME_SHA256" =~ ^[0-9a-f]{64}$ ]] ||
  fail "NEUFLOW_RUNTIME_SHA256 must be a lowercase SHA256"
actual_runtime_sha=$(sha256sum -- "$runtime_archive" | awk '{print $1}')
[[ "$actual_runtime_sha" == "$NEUFLOW_RUNTIME_SHA256" ]] ||
  fail "conda-pack runtime SHA256 mismatch: expected $NEUFLOW_RUNTIME_SHA256, got $actual_runtime_sha"

# Runtime legal review: exact bytes + reviewed:true bound to the generated inventory hash.
[[ "$(stat -c '%a' "$runtime_legal_review_file")" == 644 ]] ||
  fail "runtime legal-review input must have mode 0644: $runtime_legal_review_file"
[[ "$NEUFLOW_RUNTIME_LEGAL_REVIEW_SHA256" =~ ^[0-9a-f]{64}$ ]] ||
  fail "NEUFLOW_RUNTIME_LEGAL_REVIEW_SHA256 must be a lowercase SHA256"
actual_review_sha=$(sha256sum -- "$runtime_legal_review_file" | awk '{print $1}')
[[ "$actual_review_sha" == "$NEUFLOW_RUNTIME_LEGAL_REVIEW_SHA256" ]] ||
  fail "runtime legal-review SHA256 mismatch: expected $NEUFLOW_RUNTIME_LEGAL_REVIEW_SHA256, got $actual_review_sha"

runtime_legal_inventory="$runtime_legal_dir/runtime-license-inventory.json"
require_regular "$runtime_legal_dir/RUNTIME-LICENSES.txt" "NeuFlow-validation runtime license aggregate"
require_regular "$runtime_legal_dir/RUNTIME-NOTICES.txt" "NeuFlow-validation runtime notice aggregate"
require_regular "$runtime_legal_inventory" "NeuFlow-validation runtime license inventory"
runtime_inventory_sha=$(sha256sum -- "$runtime_legal_inventory" | awk '{print $1}')
python3.11 "$root/tools/p25_5/licenses.py" validate-runtime-review \
  --input "$runtime_legal_review_file" \
  --sha256 "$NEUFLOW_RUNTIME_LEGAL_REVIEW_SHA256" \
  --inventory-sha256 "$runtime_inventory_sha"
# Recheck the exact relocated runtime that will actually be carried against the approved inventory.
python3.11 "$root/tools/p25_5/licenses.py" verify-runtime-content \
  --prefix "$runtime_root" \
  --inventory "$runtime_legal_inventory" \
  --inventory-sha256 "$runtime_inventory_sha"

# Assemble the staged package tree. Every carried file except the wrapper is 0644; the wrapper is
# the only 0755 regular file (mirrors the P25-6/WAFT package contract). NO upstream source tree is
# staged (see the header): the operator supplies their own checkout.
staging_root="$output/staging"
pkg="$staging_root/whitewater-neuflow-validation-el8"
rm -rf "$staging_root"
mkdir -p "$pkg/scripts" "$pkg/models" "$pkg/runtime" "$pkg/legal" "$pkg/bakeoff/neuflow-validation"

install -m 0755 "$wrapper" "$pkg/scripts/ww-bakeoff-airgap"
install -m 0644 "$export_entrypoint" "$pkg/models/export_neuflow_v2.py"
install -m 0644 "$artifact_workflow" "$pkg/models/artifact_workflow.py"
install -m 0644 "$exclusion_contract" "$pkg/models/exclusion_contract.py"
install -m 0644 "$manifest" "$pkg/models/neuflow-v2.json"
install -m 0644 "$runtime_archive" "$pkg/runtime/whitewater-neuflow-validation-runtime.tar.gz"
install -m 0644 "$runtime_sha_file" "$pkg/runtime/whitewater-neuflow-validation-runtime.tar.gz.sha256"
install -m 0644 "$runtime_inventory" "$pkg/runtime/whitewater-neuflow-validation-runtime.inventory"
install -m 0644 "$runtime_legal_dir/RUNTIME-LICENSES.txt" "$pkg/legal/RUNTIME-LICENSES.txt"
install -m 0644 "$runtime_legal_dir/RUNTIME-NOTICES.txt" "$pkg/legal/RUNTIME-NOTICES.txt"
install -m 0644 "$runtime_legal_inventory" "$pkg/legal/runtime-license-inventory.json"
install -m 0644 "$runtime_legal_review_file" "$pkg/legal/runtime-legal-review.json"
install -m 0644 "$runtime_inputs" "$pkg/bakeoff/neuflow-validation/runtime-inputs.json"
install -m 0644 "$run_instructions_source" "$pkg/RUN-NEUFLOW-VALIDATION.txt"

# Hard stop: no upstream NeuFlow source tree may be present in the staged package. This pack
# deliberately vendors no source (the in-repo checkpoints make a clean vendored checkout unsafe);
# fail closed if a future change ever stages one.
if find "$pkg" -type d -name 'NeuFlow' -print -quit | grep -q .; then
  fail "staged package carries a NeuFlow/ source tree; this pack must not vendor the upstream source"
fi
if [ -e "$pkg/.git" ] || [ -n "$(find "$pkg" -type d -name '.git' -print -quit)" ]; then
  fail "staged package carries a .git checkout; no upstream source is vendored in this pack"
fi

# Hard stop: the checkpoint must never be inside the staged tree.
if find "$pkg" -type f \( -name '*.pth' -o -name 'neuflow_*.pth' \) -print -quit | grep -q .; then
  fail "a checkpoint-like file is present in the staged package; the checkpoint must never be carried"
fi
# Belt-and-braces: no file in the staged tree may match the pinned checkpoint bytes.
while IFS= read -r -d '' f; do
  [[ "$(stat -c '%s' "$f")" == "$NEUFLOW_CHECKPOINT_SIZE" ]] || continue
  if [[ "$(sha256sum -- "$f" | awk '{print $1}')" == "$NEUFLOW_CHECKPOINT_SHA256" ]]; then
    fail "the pinned NeuFlow checkpoint bytes are present in the staged package: $f"
  fi
done < <(find "$pkg" -type f -print0)

package_tarball="$output/whitewater-neuflow-validation-el8.tar.gz"
package_checksum="$package_tarball.sha256"
echo "NeuFlow-validation: assembling outer archive from the staged tree"
tar -C "$staging_root" -czf "$package_tarball" "whitewater-neuflow-validation-el8"
package_sha=$(sha256sum -- "$package_tarball" | awk '{print $1}')
printf '%s  %s\n' "$package_sha" "$(basename "$package_tarball")" > "$package_checksum"
chmod 0644 "$package_tarball" "$package_checksum"

require_regular "$package_tarball" "NeuFlow-validation package tarball"
require_regular "$package_checksum" "NeuFlow-validation package SHA256"

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
        raise SystemExit("NeuFlow-validation package archive is empty")
    for member in members:
        if member.issym() or member.islnk():
            raise SystemExit(f"NeuFlow-validation package archive contains a link: {member.name}")
        if not member.isdir() and not member.isfile():
            raise SystemExit(f"NeuFlow-validation package archive contains a non-regular entry: {member.name}")
        if member.isfile():
            name = PurePosixPath(member.name)
            mode = stat.S_IMODE(member.mode)
            if name.name == "ww-bakeoff-airgap":
                if mode != 0o755:
                    raise SystemExit(f"{member.name}: the airgap wrapper must be 0755")
                continue
            if mode not in (0o644, 0o755):
                raise SystemExit(f"{member.name}: carried file has unexpected mode {mode:04o}")
print(f"NeuFlow-validation package archive structure: PASS ({len(members)} entries; no links)")
PY

cat > "$output/NEUFLOW-VALIDATION.txt" <<'EOF'
White Water NeuFlow v2 export/validation runtime package
========================================================

This archive lets the operator run models/export_neuflow_v2.py OFFLINE on the airgapped EL8 box to
re-export and qualify the pinned NeuFlow v2 candidate as a linux-x86_64 ONNX artifact. It carries
the conda-pack export/validation runtime and the export entrypoint plus its models/ closure. It
carries NO upstream source (NeuFlow v2 tracks its checkpoints in-repo, so no clean checkout can be
vendored without redistributing the unknown-licence checkpoint) and NO checkpoint: the operator
supplies their own clean NeuFlow_v2 checkout at the pinned commit and verifies the checkpoint's
pinned SHA256/size. It selects no model and authorizes no redistribution of the checkpoint or
source. Follow RUN-NEUFLOW-VALIDATION.txt. No production images or checkpoints leave the machine.
EOF
chmod 0644 "$output/NEUFLOW-VALIDATION.txt"

if ! grep -Eiq 'neuflow_mixed|checkpoint|export_neuflow_v2' "$run_instructions_source"; then
  fail "run instructions do not describe the NeuFlow export/validation procedure"
fi

echo "NeuFlow-validation CI: qualification and package passed"
echo "  tarball: $package_tarball"
echo "  sha256:  $package_sha"
echo "  NeuFlow source commit (operator-supplied, not carried): $NEUFLOW_SOURCE_COMMIT"
