#!/usr/bin/env bash
#
# The fail-closed seam between the EL8 CI job and the P25-6 target-measurement packager.
#
# P25-6 carries the resumable profile driver (tools/bakeoff/run.py) and a runtime that adds the
# OpenEXR Python bindings and pynvml on top of the P25-5 evaluator base. This seam reuses the exact
# P25-5 packager (tools/p25_5/package.py) and legal tooling unchanged; it differs only in the
# carried module closure, the runtime it qualifies, and an extra driver/OpenEXR/pynvml import smoke.
#
# The candidate CPU-correctness gate is unchanged from P25-5: the carried evaluator.py `verify`
# is run once per carried candidate artifact against the CPU provider. The driver itself is then
# import-smoked in the relocated runtime so a broken module closure or a missing OpenEXR/pynvml
# dependency fails here rather than on the airgapped box.
#
# Required environment (mirrors ci-p25-5-qualify.sh with P25_6_ names):
#   P25_6_ADMISSION_FILE       v2 candidate-admission JSON
#   P25_6_LEGAL_REVIEW_FILE    exact operator legal-review JSON used to create admission
#   P25_6_LEGAL_REVIEW_SHA256  SHA256 of that exact operator legal-review JSON
#   P25_6_CANDIDATE_LICENSE_INPUT  checked-in candidate license/notice declaration
#   P25_6_RUNTIME_LICENSE_INPUT    runtime license/notice declaration bound to the explicit lock
#   P25_6_RUNTIME_LEGAL_DIR        generated runtime license/notice bundle directory
#   P25_6_RUNTIME_LEGAL_REVIEW_FILE/SHA256  Andrew Miller approval bound to generated inventory
#   P25_6_RUNTIME_ROOT         extracted, conda-unpacked runtime used by the evaluator/driver
#   P25_6_RUNTIME_ARCHIVE      exact conda-pack archive
#   P25_6_RUNTIME_SHA256       SHA256 of that archive
#   P25_6_RUNTIME_SHA256_FILE  checksum file for the archive
#   P25_6_RUNTIME_INVENTORY    explicit conda package inventory
#   P25_6_RUNTIME_CLOSURE      ORT dependency-closure audit
#   P25_6_QUALIFIER            Python evaluator entry point (verify --manifest ... --provider cpu)
#   P25_6_DRIVER               carried profile-driver module path (tools/bakeoff/run.py)
#   P25_6_PACKAGER             Python package.py-compatible packager entry point
#   P25_6_PACKAGE_SPEC         package-spec template naming every carried source file; its one
#                              runtime record uses the __P25_6_RUNTIME_* placeholders
#   P25_6_RUN_INSTRUCTIONS     explicit target-measurement run instructions source (RUN-P25-6.txt)
#   P25_6_OUTPUT_DIR           output directory for the carried package/evidence
#
# P25_6_PACKAGE_SPEC is a template. Its admission object must set
# candidates=__P25_6_ADMISSION_CANDIDATES__, and it must contain exactly one runtime file record
# with role=runtime, candidate_id=null, mode=0644, source=__P25_6_RUNTIME_ARCHIVE__,
# sha256=__P25_6_RUNTIME_SHA256__, and size_bytes=__P25_6_RUNTIME_SIZE_BYTES__. After
# qualification, this script materializes the generated admission candidates, runtime identity
# fields and actual local source paths in a spec beside the template; every other field is
# retained byte-for-byte and remains subject to package.py's schema/admission checks.

set -euo pipefail

fail() {
  echo "P25-6 CI: $*" >&2
  exit 1
}

require_env() {
  local name="$1"
  [[ -n "${!name:-}" ]] || fail "required input $name is empty; P25-6 is fail-closed"
}

require_regular() {
  local path="$1"
  local label="$2"
  [[ -e "$path" ]] || fail "$label is missing: $path"
  [[ ! -L "$path" ]] || fail "$label must not be a symlink: $path"
  [[ -f "$path" ]] || fail "$label must be a regular file: $path"
}

require_directory() {
  local path="$1"
  local label="$2"
  [[ -d "$path" ]] || fail "$label is missing: $path"
  [[ ! -L "$path" ]] || fail "$label must not be a symlink: $path"
}

for required in \
  P25_6_ADMISSION_FILE \
  P25_6_LEGAL_REVIEW_FILE \
  P25_6_LEGAL_REVIEW_SHA256 \
  P25_6_CANDIDATE_LICENSE_INPUT \
  P25_6_RUNTIME_LICENSE_INPUT \
  P25_6_RUNTIME_LEGAL_DIR \
  P25_6_RUNTIME_LEGAL_REVIEW_FILE \
  P25_6_RUNTIME_LEGAL_REVIEW_SHA256 \
  P25_6_RUNTIME_ROOT \
  P25_6_RUNTIME_ARCHIVE \
  P25_6_RUNTIME_SHA256 \
  P25_6_RUNTIME_SHA256_FILE \
  P25_6_RUNTIME_INVENTORY \
  P25_6_RUNTIME_CLOSURE \
  P25_6_QUALIFIER \
  P25_6_DRIVER \
  P25_6_PACKAGER \
  P25_6_PACKAGE_SPEC \
  P25_6_RUN_INSTRUCTIONS \
  P25_6_OUTPUT_DIR; do
  require_env "$required"
done

root="${GITHUB_WORKSPACE:-$(pwd)}"
output="$P25_6_OUTPUT_DIR"
mkdir -p "$output"

resolve_repo_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s/%s\n' "$root" "$1" ;;
  esac
}

copy_if_needed() {
  local source="$1"
  local destination="$2"
  if [[ -e "$destination" ]] && cmp -- "$source" "$destination" >/dev/null 2>&1; then
    return 0
  fi
  cp -- "$source" "$destination"
}

admission=$(resolve_repo_path "$P25_6_ADMISSION_FILE")
legal_review_file=$(resolve_repo_path "$P25_6_LEGAL_REVIEW_FILE")
candidate_license_input=$(resolve_repo_path "$P25_6_CANDIDATE_LICENSE_INPUT")
runtime_license_input=$(resolve_repo_path "$P25_6_RUNTIME_LICENSE_INPUT")
runtime_legal_review_file=$(resolve_repo_path "$P25_6_RUNTIME_LEGAL_REVIEW_FILE")
runtime_legal_dir=$(resolve_repo_path "$P25_6_RUNTIME_LEGAL_DIR")
qualifier=$(resolve_repo_path "$P25_6_QUALIFIER")
driver=$(resolve_repo_path "$P25_6_DRIVER")
packager=$(resolve_repo_path "$P25_6_PACKAGER")
package_spec=$(resolve_repo_path "$P25_6_PACKAGE_SPEC")
run_instructions_source=$(resolve_repo_path "$P25_6_RUN_INSTRUCTIONS")
runtime_root=$(resolve_repo_path "$P25_6_RUNTIME_ROOT")
runtime_archive=$(resolve_repo_path "$P25_6_RUNTIME_ARCHIVE")
runtime_sha_file=$(resolve_repo_path "$P25_6_RUNTIME_SHA256_FILE")
runtime_inventory=$(resolve_repo_path "$P25_6_RUNTIME_INVENTORY")
runtime_closure=$(resolve_repo_path "$P25_6_RUNTIME_CLOSURE")

require_regular "$admission" "P25-6 admission file"
require_regular "$legal_review_file" "P25-6 legal-review file"
require_regular "$candidate_license_input" "P25-6 candidate license input"
require_regular "$runtime_license_input" "P25-6 runtime license input"
require_regular "$runtime_legal_review_file" "P25-6 runtime legal-review file"
require_directory "$runtime_legal_dir" "P25-6 generated runtime legal directory"
require_regular "$qualifier" "P25-6 qualifier"
require_regular "$driver" "P25-6 profile driver"
require_regular "$packager" "P25-6 packager"
require_regular "$package_spec" "P25-6 package specification"
require_regular "$run_instructions_source" "P25-6 run instructions source"
require_directory "$runtime_root" "P25-6 extracted runtime"
require_regular "$runtime_archive" "P25-6 conda-pack runtime archive"
require_regular "$runtime_sha_file" "P25-6 runtime SHA256 file"
require_regular "$runtime_inventory" "P25-6 runtime package inventory"
require_regular "$runtime_closure" "P25-6 runtime dependency-closure audit"

# Loader path for every invocation of the relocated runtime's own python below.  Prepend the
# runtime's lib so the conda libstdc++ (CXXABI_1.3.15, required by the OpenEXR/Imath native .so
# the OpenEXR Python bindings load) resolves ahead of the EL8 system /lib64/libstdc++.so.6, which
# lacks it; the .cpython RUNPATH
# does not propagate to transitively-loaded libraries, so RUNPATH alone loses that race.  We
# scope this per-invocation (not a global export) so it never reaches the system python3.11 or
# other host tools this script also runs, and we compose with any pre-existing value rather than
# clobber it, mirroring the ww-bakeoff-airgap wrapper.
runtime_ld_library_path="$runtime_root/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

[[ "$(stat -c '%a' "$legal_review_file")" == 644 ]] ||
  fail "P25-6 legal-review file must have mode 0644: $legal_review_file"
[[ "$P25_6_LEGAL_REVIEW_SHA256" =~ ^[0-9a-f]{64}$ ]] ||
  fail "P25_6_LEGAL_REVIEW_SHA256 must be a lowercase SHA256"
actual_legal_review_sha=$(sha256sum -- "$legal_review_file" | awk '{print $1}')
[[ "$actual_legal_review_sha" == "$P25_6_LEGAL_REVIEW_SHA256" ]] ||
  fail "legal-review SHA256 mismatch: expected $P25_6_LEGAL_REVIEW_SHA256, got $actual_legal_review_sha"

for legal_input in "$candidate_license_input" "$runtime_license_input" "$runtime_legal_review_file"; do
  [[ "$(stat -c '%a' "$legal_input")" == 644 ]] ||
    fail "license/review input must have mode 0644: $legal_input"
done

[[ "$P25_6_RUNTIME_SHA256" =~ ^[0-9a-f]{64}$ ]] ||
  fail "P25_6_RUNTIME_SHA256 must be a lowercase SHA256"
actual_runtime_sha=$(sha256sum -- "$runtime_archive" | awk '{print $1}')
[[ "$actual_runtime_sha" == "$P25_6_RUNTIME_SHA256" ]] ||
  fail "conda-pack runtime SHA256 mismatch: expected $P25_6_RUNTIME_SHA256, got $actual_runtime_sha"

# The admission document is independent of the eventual evaluator/driver CLI. It uses the v2
# report candidate vocabulary so an excluded-but-measurable candidate remains available for
# comparison, while unavailable/non-measurable candidates can never enter the package.
admitted_ids_file="$output/.p25-6-admitted-candidate-ids"
python3 - "$admission" "$admitted_ids_file" "$P25_6_LEGAL_REVIEW_SHA256" <<'PY'
import json
import re
import stat
import sys
from pathlib import Path

admission_path = Path(sys.argv[1])
ids_path = Path(sys.argv[2])
expected_legal_review_sha = sys.argv[3]
try:
    document = json.loads(admission_path.read_text(encoding="utf-8"))
except (OSError, ValueError) as exc:
    raise SystemExit(f"invalid admission JSON: {exc}")

if not isinstance(document, dict) or document.get("protocol_id") != "whitewater-p25-v2":
    raise SystemExit("admission file must be a whitewater-p25-v2 document")
if document.get("legal_review_sha256") != expected_legal_review_sha:
    raise SystemExit(
        "admission legal_review_sha256 does not match the operator-supplied legal-review input"
    )
candidates = document.get("candidates")
if not isinstance(candidates, list) or not candidates:
    raise SystemExit("admission file must contain a non-empty candidates array")

sha256 = re.compile(r"^[0-9a-f]{64}$")
commit = re.compile(r"^[0-9a-f]{40}$")
ids = []
for index, candidate in enumerate(candidates):
    path = f"candidates[{index}]"
    if not isinstance(candidate, dict):
        raise SystemExit(f"{path} must be an object")
    candidate_id = candidate.get("candidate_id")
    if not isinstance(candidate_id, str) or not candidate_id:
        raise SystemExit(f"{path}.candidate_id must be a non-empty string")
    if candidate_id in ids:
        raise SystemExit(f"duplicate admitted candidate: {candidate_id}")
    ids.append(candidate_id)
    if candidate.get("measurement_status") != "measurable":
        raise SystemExit(
            f"{candidate_id}: measurement_status must be measurable; refusing unavailable candidate"
        )
    if candidate.get("status") not in {"eligible", "excluded"}:
        raise SystemExit(f"{candidate_id}: status must be eligible or excluded")
    if candidate.get("status") == "excluded" and not isinstance(candidate.get("exclusion_reason"), dict):
        raise SystemExit(f"{candidate_id}: excluded candidate needs an explicit exclusion_reason")
    for field, pattern in (
        ("source_commit", commit),
        ("checkpoint_sha256", sha256),
        ("artifact_sha256", sha256),
        ("export_environment_sha256", sha256),
        ("manifest_sha256", sha256),
    ):
        value = candidate.get(field)
        if not isinstance(value, str) or pattern.fullmatch(value) is None:
            raise SystemExit(f"{candidate_id}: {field} is missing or not an exact lowercase hash")
    size = candidate.get("artifact_size_bytes")
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise SystemExit(f"{candidate_id}: artifact_size_bytes must be positive")
    providers = candidate.get("measurement_providers")
    if not isinstance(providers, list) or "cpu" not in providers or len(set(providers)) != len(providers):
        raise SystemExit(f"{candidate_id}: measurable candidate must explicitly qualify CPU")

ids_path.write_text("\n".join(sorted(ids)) + "\n", encoding="utf-8")
ids_path.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)
print(f"P25-6 admission: {len(ids)} measurable candidate(s); shipping selection remains unset")
for candidate_id in sorted(ids):
    print(f"  {candidate_id}")
PY

copy_if_needed "$admission" "$output/admission.json"
copy_if_needed "$legal_review_file" "$output/legal-review.json"
printf '%s  %s\n' "$P25_6_LEGAL_REVIEW_SHA256" "legal-review.json" > "$output/legal-review.json.sha256"
chmod 0644 "$output/legal-review.json.sha256"
copy_if_needed "$admitted_ids_file" "$output/admitted-candidates.txt"
copy_if_needed "$runtime_archive" "$output/whitewater-p25-6-runtime.tar.gz"
copy_if_needed "$runtime_sha_file" "$output/whitewater-p25-6-runtime.tar.gz.sha256"
copy_if_needed "$runtime_inventory" "$output/whitewater-p25-6-runtime.inventory"
copy_if_needed "$runtime_closure" "$output/whitewater-p25-6-runtime-closure.txt"
copy_if_needed "$package_spec" "$output/package-spec-template.json"

# Candidate evidence is generated from the checked-in declaration only after the fresh export
# and admission inputs are present. It is a reproducible aggregate, not a legal conclusion.
candidate_legal_dir="$output/legal-candidate"
python3 "$root/tools/p25_5/licenses.py" candidate \
  --input "$candidate_license_input" \
  --output-dir "$candidate_legal_dir" \
  --license-name SEA-RAFT-LICENSE.txt \
  --notice-name SEA-RAFT-NOTICE.txt
require_regular "$candidate_legal_dir/SEA-RAFT-LICENSE.txt" "P25-6 candidate license aggregate"
require_regular "$candidate_legal_dir/SEA-RAFT-NOTICE.txt" "P25-6 candidate notice aggregate"
require_regular "$candidate_legal_dir/candidate-license-inventory.json" "P25-6 candidate license inventory"
copy_if_needed "$candidate_license_input" "$output/candidate-license-input.json"

# The workflow creates this directory from the final conda-pack extraction. Recheck the exact
# generated files and the human approval here so a future caller cannot bypass the workflow gate
# by invoking this seam directly.
require_regular "$runtime_legal_dir/RUNTIME-LICENSES.txt" "P25-6 runtime license aggregate"
require_regular "$runtime_legal_dir/RUNTIME-NOTICES.txt" "P25-6 runtime notice aggregate"
runtime_legal_inventory="$runtime_legal_dir/runtime-license-inventory.json"
require_regular "$runtime_legal_inventory" "P25-6 runtime license inventory"
runtime_inventory_sha=$(sha256sum -- "$runtime_legal_inventory" | awk '{print $1}')
if [[ -n "${P25_6_RUNTIME_LEGAL_INVENTORY_SHA256:-}" ]]; then
  [[ "$P25_6_RUNTIME_LEGAL_INVENTORY_SHA256" == "$runtime_inventory_sha" ]] ||
    fail "runtime legal inventory SHA256 mismatch: workflow=$P25_6_RUNTIME_LEGAL_INVENTORY_SHA256 actual=$runtime_inventory_sha"
fi
python3 "$root/tools/p25_5/licenses.py" validate-runtime-review \
  --input "$runtime_legal_review_file" \
  --sha256 "$P25_6_RUNTIME_LEGAL_REVIEW_SHA256" \
  --inventory-sha256 "$runtime_inventory_sha"
python3 "$root/tools/p25_5/licenses.py" verify-runtime-content \
  --prefix "$runtime_root" \
  --inventory "$runtime_legal_inventory" \
  --inventory-sha256 "$runtime_inventory_sha"
copy_if_needed "$runtime_license_input" "$output/runtime-license-input.json"
copy_if_needed "$runtime_legal_review_file" "$output/runtime-legal-review.json"

qualification_dir="$output/qualification"
mkdir -p "$qualification_dir"

# Derive the exact worklist from the operator-supplied package-spec template. Every technically
# measurable candidate must have one manifest and at least one carried artifact.
qualification_inputs="$output/.p25-6-qualification-inputs.tsv"
python3 - "$package_spec" "$admitted_ids_file" "$qualification_inputs" <<'PY'
import json
import os
import re
import stat
import sys
from pathlib import Path

spec_path = Path(sys.argv[1]).resolve()
ids_path = Path(sys.argv[2])
inputs_path = Path(sys.argv[3])
try:
    document = json.loads(spec_path.read_text(encoding="utf-8"))
except (OSError, UnicodeError, ValueError) as exc:
    raise SystemExit(f"invalid P25-6 package-spec template: {exc}")
if not isinstance(document, dict) or not isinstance(document.get("files"), list):
    raise SystemExit("P25-6 package-spec template must contain a files array")
ids = [line.strip() for line in ids_path.read_text(encoding="utf-8").splitlines() if line.strip()]
if not ids:
    raise SystemExit("P25-6 qualification worklist is empty")
candidate_id_re = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
if any(candidate_id_re.fullmatch(candidate_id) is None for candidate_id in ids):
    raise SystemExit("P25-6 candidate IDs must be safe package path tokens")
admitted = set(ids)
manifests = {candidate_id: [] for candidate_id in ids}
artifacts = {candidate_id: [] for candidate_id in ids}

for index, item in enumerate(document["files"]):
    if not isinstance(item, dict):
        raise SystemExit(f"package-spec files[{index}] must be an object")
    role = item.get("role")
    if role not in {"candidate-manifest", "model-artifact"}:
        continue
    candidate_id = item.get("candidate_id")
    if not isinstance(candidate_id, str) or candidate_id not in admitted:
        raise SystemExit(
            f"package-spec files[{index}] binds {role} to a candidate outside explicit admission"
        )
    source = item.get("source")
    if not isinstance(source, str) or not source or "://" in source:
        raise SystemExit(f"package-spec files[{index}].source must be a local path")
    source_path = Path(source)
    if not source_path.is_absolute():
        source_path = spec_path.parent / source_path
    source_path = source_path.resolve()
    try:
        info = source_path.lstat()
    except OSError as exc:
        raise SystemExit(f"package-spec files[{index}] source is missing: {source_path}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise SystemExit(f"package-spec files[{index}] source must be a regular non-symlink file")
    if stat.S_IMODE(info.st_mode) != 0o644:
        raise SystemExit(f"package-spec files[{index}] source must have mode 0644: {source_path}")
    if role == "candidate-manifest":
        manifests[candidate_id].append(source_path)
    else:
        artifacts[candidate_id].append(source_path)

lines = []
for candidate_id in ids:
    if len(manifests[candidate_id]) != 1:
        raise SystemExit(
            f"{candidate_id}: package-spec must carry exactly one candidate-manifest source"
        )
    if not artifacts[candidate_id]:
        raise SystemExit(f"{candidate_id}: package-spec must carry at least one model-artifact source")
    manifest = manifests[candidate_id][0]
    for artifact in sorted(artifacts[candidate_id], key=str):
        lines.append(f"{candidate_id}\t{manifest}\t{artifact}")
inputs_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
os.chmod(inputs_path, 0o644)
print(f"P25-6 qualification worklist: {len(lines)} manifest/artifact pair(s)")
PY

echo "P25-6: invoking evaluator.py verify for every carried candidate artifact (CPU path)"
export P25_6_QUALIFICATION_ADMISSION_FILE="$admission"
export P25_6_QUALIFICATION_RUNTIME_ROOT="$runtime_root"
export P25_6_QUALIFICATION_PROVIDER="cpu"
qualification_index=0
while IFS=$'\t' read -r candidate_id manifest artifact; do
  [[ -n "$candidate_id" && -n "$manifest" && -n "$artifact" ]] ||
    fail "P25-6 qualification worklist contains an incomplete row"
  result_file="$qualification_dir/${candidate_id}-${qualification_index}.verify.json"
  echo "  $candidate_id: verify $(basename "$artifact")"
  LD_LIBRARY_PATH="$runtime_ld_library_path" "$runtime_root/bin/python" "$qualifier" verify \
    --manifest "$manifest" \
    --artifact "$artifact" \
    --protocol "$root/bakeoff/protocol-v2.json" \
    --provider cpu \
    > "$result_file"
  chmod 0644 "$result_file"
  qualification_index=$((qualification_index + 1))
done < "$qualification_inputs"
(( qualification_index > 0 )) || fail "P25-6 evaluator produced no qualification results"

# Driver + OpenEXR + pynvml import smoke in the relocated runtime. This is the P25-6-specific gate:
# the driver's whole first-party module closure must import, and the two production/measurement
# dependencies the P25-5 runtime lacked (the OpenEXR Python bindings, pynvml) must be present. It
# runs no inference and touches no production data.
echo "P25-6: driver/OpenEXR/pynvml import smoke in the relocated runtime"
PYTHONPATH="$root" LD_LIBRARY_PATH="$runtime_ld_library_path" "$runtime_root/bin/python" - "$driver" <<'PY'
import importlib
import sys
from pathlib import Path

driver_path = Path(sys.argv[1]).resolve()
# Import the driver and its full closure through the package path, exactly as the airgap wrapper
# launches it (-m tools.bakeoff.run with the package root on PYTHONPATH).
run = importlib.import_module("tools.bakeoff.run")
if not hasattr(run, "main") or not hasattr(run, "run_bakeoff"):
    raise SystemExit("carried driver does not expose the expected entry points")
# The production input and NVML backends must be importable in this runtime.
import OpenEXR  # noqa: F401
import pynvml  # noqa: F401
# The driver's CLI parser must build (argparse construction only; no run).
run._parser()
print("P25-6 driver/OpenEXR/pynvml import smoke: PASS")
print(f"  driver: {driver_path}")
print(f"  OpenEXR: {OpenEXR.__file__}")
print(f"  pynvml: {pynvml.__file__}")
PY

# Materialize the carried candidate/artifact-map identity from the freshly exported linux manifest,
# in place, immediately before packaging (Finding A). The checked-in inputs are platform-neutral
# PLACEHOLDER templates; filling them here binds the linux-x86_64 artifact_sha256/manifest_sha256/
# export_environment_sha256/artifact_size_bytes and artifact-map platform from the generated
# models/sea-raft-m.json -- never a hardcoded, reproducible-forever constant -- exactly as the
# admission document is generated after export. The linux export manifest is already in the
# workspace at this point (the p25-6 lane restores it before this seam runs). Without this, the
# packaged inputs would carry the checked-in macOS binding and the driver's
# validate_manifest_artifact would raise artifact_hash_mismatch against the packaged linux ONNX
# before any profile runs. A non-linux manifest here fails closed inside the materializer.
inputs_dir="${package_spec%/*}/inputs"
generated_candidate_manifest="$root/models/sea-raft-m.json"
require_regular "$generated_candidate_manifest" "P25-6 generated candidate manifest"
require_regular "$inputs_dir/candidate-entries.json" "P25-6 carried candidate-entries template"
require_regular "$inputs_dir/artifact-map.json" "P25-6 carried artifact-map template"
python3 "$root/tools/p25_5/p25_6_materialize_inputs.py" \
  --manifest "$generated_candidate_manifest" \
  --candidate-entries "$inputs_dir/candidate-entries.json" \
  --artifact-map "$inputs_dir/artifact-map.json"

# Materialize a fresh spec after qualification, immediately before invoking package.py. This
# patches only the tightly defined admission and runtime placeholders; candidate files are never
# inferred from a directory listing.
generated_package_spec="${package_spec%/*}/.whitewater-p25-6-generated-package-spec.json"
python3 - "$package_spec" "$generated_package_spec" "$runtime_archive" "$P25_6_RUNTIME_SHA256" "$admission" \
  "$candidate_legal_dir" "$runtime_legal_dir" "$runtime_legal_review_file" <<'PY'
import json
import os
import re
import stat
import sys
from pathlib import Path

template_path = Path(sys.argv[1])
generated_path = Path(sys.argv[2])
runtime_path = Path(sys.argv[3])
runtime_sha = sys.argv[4]
admission_path = Path(sys.argv[5])
candidate_legal_dir = Path(sys.argv[6])
runtime_legal_dir = Path(sys.argv[7])
runtime_review_path = Path(sys.argv[8])

try:
    document = json.loads(template_path.read_text(encoding="utf-8"))
except (OSError, UnicodeError, ValueError) as exc:
    raise SystemExit(f"invalid P25-6 package-spec template: {exc}")
if not isinstance(document, dict):
    raise SystemExit("P25-6 package-spec template must be a JSON object")
try:
    admission_document = json.loads(admission_path.read_text(encoding="utf-8"))
except (OSError, UnicodeError, ValueError) as exc:
    raise SystemExit(f"invalid generated P25-6 admission document: {exc}")
if not isinstance(admission_document, dict) or admission_document.get("protocol_id") != "whitewater-p25-v2":
    raise SystemExit("generated P25-6 admission must identify whitewater-p25-v2")
raw_admission_candidates = admission_document.get("candidates")
if not isinstance(raw_admission_candidates, list) or not raw_admission_candidates:
    raise SystemExit("generated P25-6 admission must contain a non-empty candidates array")
package_candidates = []
candidate_id_re = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
seen_candidate_ids = set()
for index, candidate in enumerate(raw_admission_candidates):
    if not isinstance(candidate, dict):
        raise SystemExit(f"generated admission candidates[{index}] must be an object")
    candidate_id = candidate.get("candidate_id")
    if not isinstance(candidate_id, str) or candidate_id_re.fullmatch(candidate_id) is None:
        raise SystemExit(f"generated admission candidates[{index}] has an unsafe candidate_id")
    if candidate_id in seen_candidate_ids:
        raise SystemExit(f"generated admission contains duplicate candidate_id: {candidate_id}")
    seen_candidate_ids.add(candidate_id)
    if candidate.get("measurement_status") != "measurable":
        raise SystemExit(f"{candidate_id}: non-measurable candidates cannot enter package admission")
    if candidate.get("measurement_admitted") is not True:
        raise SystemExit(f"{candidate_id}: admission must explicitly set measurement_admitted=true")
    status = candidate.get("status")
    if status not in {"eligible", "excluded"}:
        raise SystemExit(f"{candidate_id}: generated admission has invalid shipping status {status!r}")
    package_candidate = {
        "candidate_id": candidate_id,
        "measurement_status": "measurable",
        "measurement_admitted": True,
        "status": status,
    }
    if status == "excluded":
        reason = candidate.get("exclusion_reason")
        if isinstance(reason, dict):
            message = reason.get("message")
            if not isinstance(message, str) or not message:
                message = json.dumps(reason, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        elif isinstance(reason, str) and reason:
            message = reason
        else:
            raise SystemExit(f"{candidate_id}: excluded admission needs an exclusion_reason")
        package_candidate["exclusion_reason"] = message
    package_candidates.append(package_candidate)
admission = document.get("admission")
if not isinstance(admission, dict) or admission.get("candidates") != "__P25_6_ADMISSION_CANDIDATES__":
    raise SystemExit(
        "P25-6 package-spec template admission.candidates must be exactly "
        "__P25_6_ADMISSION_CANDIDATES__"
    )
admission["candidates"] = package_candidates
files = document.get("files")
if not isinstance(files, list):
    raise SystemExit("P25-6 package-spec template must contain a files array")
runtime_entries = [item for item in files if isinstance(item, dict) and item.get("role") == "runtime"]
if len(runtime_entries) != 1:
    raise SystemExit("P25-6 package-spec template must contain exactly one runtime file")
runtime = runtime_entries[0]
expected = {
    "candidate_id": None,
    "mode": "0644",
    "source": "__P25_6_RUNTIME_ARCHIVE__",
    "sha256": "__P25_6_RUNTIME_SHA256__",
    "size_bytes": "__P25_6_RUNTIME_SIZE_BYTES__",
}
for key, value in expected.items():
    if runtime.get(key) != value:
        raise SystemExit(
            f"runtime placeholder {key!r} must be exactly {value!r}; got {runtime.get(key)!r}"
        )
if not runtime_path.exists() or runtime_path.is_symlink() or not runtime_path.is_file():
    raise SystemExit(f"P25-6 runtime archive is not a local regular file: {runtime_path}")
runtime_stat = runtime_path.stat()
if stat.S_IMODE(runtime_stat.st_mode) != 0o644:
    raise SystemExit(
        f"P25-6 runtime archive must have mode 0644 for package.py: "
        f"{runtime_path} has {stat.S_IMODE(runtime_stat.st_mode):04o}"
    )
if runtime_stat.st_size <= 0:
    raise SystemExit("P25-6 runtime archive must be non-empty")
if re.fullmatch(r"[0-9a-f]{64}", runtime_sha) is None:
    raise SystemExit("P25_6_RUNTIME_SHA256 must be a lowercase 64-hex digest")

runtime["source"] = str(runtime_path.resolve())
runtime["sha256"] = runtime_sha
runtime["size_bytes"] = runtime_stat.st_size

bindings = {
    "__P25_6_CANDIDATE_LICENSE_SOURCE__": candidate_legal_dir / "SEA-RAFT-LICENSE.txt",
    "__P25_6_CANDIDATE_NOTICE_SOURCE__": candidate_legal_dir / "SEA-RAFT-NOTICE.txt",
    "__P25_6_CANDIDATE_INVENTORY_SOURCE__": candidate_legal_dir / "candidate-license-inventory.json",
    "__P25_6_RUNTIME_LICENSE_SOURCE__": runtime_legal_dir / "RUNTIME-LICENSES.txt",
    "__P25_6_RUNTIME_NOTICE_SOURCE__": runtime_legal_dir / "RUNTIME-NOTICES.txt",
    "__P25_6_RUNTIME_INVENTORY_SOURCE__": runtime_legal_dir / "runtime-license-inventory.json",
    "__P25_6_RUNTIME_REVIEW_SOURCE__": runtime_review_path,
}
seen_bindings = {marker: 0 for marker in bindings}
for index, item in enumerate(files):
    if not isinstance(item, dict):
        raise SystemExit(f"P25-6 package-spec files[{index}] must be an object")
    source = item.get("source")
    if source in bindings:
        seen_bindings[source] += 1
        target = bindings[source]
        try:
            info = target.lstat()
        except OSError as exc:
            raise SystemExit(f"P25-6 legal source is missing: {target}") from exc
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise SystemExit(f"P25-6 legal source must be a regular non-symlink file: {target}")
        if stat.S_IMODE(info.st_mode) != 0o644:
            raise SystemExit(f"P25-6 legal source must have mode 0644: {target}")
        item["source"] = str(target.resolve())
for marker, count in seen_bindings.items():
    if count != 1:
        raise SystemExit(f"P25-6 package-spec placeholder {marker} must appear exactly once (got {count})")

serialized = json.dumps(document, ensure_ascii=False, sort_keys=True)
if "__P25_6_" in serialized:
    raise SystemExit("P25-6 package-spec template contains an unresolved placeholder")
if generated_path.exists() and generated_path.is_symlink():
    raise SystemExit(f"generated P25-6 package spec must not be a symlink: {generated_path}")
generated_path.write_text(serialized + "\n", encoding="utf-8")
os.chmod(generated_path, 0o644)
print(f"P25-6 generated package spec: {generated_path}")
print(f"  admitted candidates: {', '.join(sorted(seen_candidate_ids))}")
print(f"  runtime source: {runtime['source']}")
print(f"  runtime sha256: {runtime_sha}")
print(f"  runtime size:   {runtime_stat.st_size} bytes")
PY
copy_if_needed "$generated_package_spec" "$output/package-spec.json"

package_tarball="$output/whitewater-p25-6-el8.tar.gz"
package_checksum="$package_tarball.sha256"
package_inventory="$output/whitewater-p25-6-el8.inventory.json"
run_instructions="$output/RUN-P25-6.txt"

echo "P25-6: invoking package builder from explicit spec"
staging_dir="$output/staging"
LD_LIBRARY_PATH="$runtime_ld_library_path" "$runtime_root/bin/python" "$packager" build \
  "$generated_package_spec" \
  --staging-dir "$staging_dir" \
  --archive "$package_tarball" \
  --inventory "$package_inventory"

package_sha=$(sha256sum -- "$package_tarball" | awk '{print $1}')
printf '%s  %s\n' "$package_sha" "$(basename "$package_tarball")" > "$package_checksum"
chmod 0644 "$package_checksum"
cp -- "$run_instructions_source" "$run_instructions"
chmod 0644 "$run_instructions"

require_regular "$package_tarball" "P25-6 package tarball"
require_regular "$package_checksum" "P25-6 package SHA256"
require_regular "$package_inventory" "P25-6 package inventory"
require_regular "$run_instructions" "P25-6 package run instructions"

expected_package_sha=$(awk 'NF { print $1; exit }' "$package_checksum")
[[ "$expected_package_sha" =~ ^[0-9a-f]{64}$ ]] ||
  fail "P25-6 package checksum file does not contain a lowercase SHA256"
actual_package_sha=$(sha256sum -- "$package_tarball" | awk '{print $1}')
[[ "$actual_package_sha" == "$expected_package_sha" ]] ||
  fail "P25-6 package SHA256 mismatch: checksum=$expected_package_sha actual=$actual_package_sha"

# The inventory is a second admission boundary after the packager.
python3 - "$package_inventory" "$admitted_ids_file" <<'PY'
import json
import sys
from pathlib import Path

inventory_path = Path(sys.argv[1])
ids_path = Path(sys.argv[2])
try:
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
except (OSError, ValueError) as exc:
    raise SystemExit(f"invalid package inventory JSON: {exc}")
if not isinstance(inventory, dict) or inventory.get("protocol_id") != "whitewater-p25-v2":
    raise SystemExit("package inventory must identify whitewater-p25-v2")
admission = inventory.get("admission")
if not isinstance(admission, dict):
    raise SystemExit("package inventory must contain an admission object")
entries = admission.get("candidates")
if not isinstance(entries, list):
    raise SystemExit("package inventory admission must contain a candidates array")
inventory_ids = []
for index, entry in enumerate(entries):
    if not isinstance(entry, dict):
        raise SystemExit(f"inventory candidates[{index}] must be an object")
    candidate_id = entry.get("candidate_id")
    if not isinstance(candidate_id, str) or not candidate_id:
        raise SystemExit(f"inventory candidates[{index}] has no candidate_id")
    if entry.get("measurement_status") != "measurable" or entry.get("measurement_admitted") is not True:
        raise SystemExit(f"{candidate_id}: packaged inventory is not explicitly measurement-admitted")
    if entry.get("status") not in {"eligible", "excluded"}:
        raise SystemExit(f"{candidate_id}: package inventory has an invalid shipping status")
    inventory_ids.append(candidate_id)
admitted_ids = [line.strip() for line in ids_path.read_text(encoding="utf-8").splitlines() if line.strip()]
if sorted(inventory_ids) != sorted(admitted_ids):
    raise SystemExit(
        "package inventory candidates differ from the explicit admission list: "
        f"admitted={sorted(admitted_ids)!r} packaged={sorted(inventory_ids)!r}"
    )
verification = inventory.get("verification")
if not isinstance(verification, dict) or any(verification.get(key) is not True for key in ("source", "staged", "archive", "extracted")):
    raise SystemExit("package inventory did not verify source/staged/archive/extracted copies")
print(f"P25-6 package inventory: {len(inventory_ids)} measurable candidate(s), measurement admission")
PY

python3 - "$package_tarball" <<'PY'
import stat
import sys
import tarfile
from pathlib import PurePosixPath

archive = sys.argv[1]
with tarfile.open(archive, "r:gz") as stream:
    members = stream.getmembers()
    if not members:
        raise SystemExit("P25-6 package archive is empty")
    for member in members:
        if member.issym() or member.islnk():
            raise SystemExit(f"P25-6 package archive contains a link: {member.name}")
        if not member.isdir() and not member.isfile():
            raise SystemExit(f"P25-6 package archive contains a non-regular entry: {member.name}")
        if member.isfile():
            suffix = PurePosixPath(member.name).suffix.lower()
            if suffix in {".onnx", ".json", ".txt", ".md", ".sha256", ".csv", ".so", ".tgz", ".gz", ".py"}:
                if stat.S_IMODE(member.mode) != 0o644:
                    raise SystemExit(f"{member.name}: carried data must have mode 0644")
print(f"P25-6 package archive structure: PASS ({len(members)} entries; no links)")
PY

cat > "$output/P25-6-TARGET-MEASUREMENT.txt" <<'EOF'
White Water Phase 2.5 P25-6 target-measurement package
======================================================

This archive drives the resumable smoke/screen/final profiles of the bake-off driver on the
airgapped Flame box. It selects no model and assigns no OFX choice index. Its candidate list is
the separately hashed, measurement-only admission list. Do not install it into Flame as a
product bundle.

After extracting on the air-gapped EL8 target, run the included conda-pack environment's
conda-unpack (the wrapper does this automatically) before invoking the driver. Follow
RUN-P25-6.txt for the exact WW_BAKEOFF_ENTRYPOINT=tools/bakeoff/run.py invocation, the three
resumable profiles, and the exactly five files the operator returns. No production images leave
the machine.
EOF
chmod 0644 "$output/P25-6-TARGET-MEASUREMENT.txt"

if ! grep -Eiq 'No production images leave|resumable|smoke' "$run_instructions"; then
  fail "run instructions do not describe the resumable target-measurement procedure"
fi

echo "P25-6 CI: qualification and target-measurement package passed"
echo "  tarball: $package_tarball"
echo "  sha256:  $expected_package_sha"
