#!/usr/bin/env python3
"""Transactional, content-validated artifacts for one bake-off run.

The runner currently writes several pieces of evidence independently.  This module provides the
smaller primitive that the coordinator can use while that work is migrated: a cell executes in a
unique attempt directory, all of its files are validated and hashed there, and one durable
``committed.json`` pointer makes that attempt visible.  A reader never searches an attempt
directory looking for a plausible result; it follows the committed pointer and validates the
complete artifact set named by that pointer.

The public layout is intentionally boring and portable::

    <parent>/<identity-sha256>/
        identity.json
        cells/<cell-sha256>/
            committed.json                 # small latest artifact_ref pointer
            manifests/<attempt-uuid>.json  # immutable, content-addressed generation
            attempts/<attempt-uuid>/...

``parent`` is supplied by the caller.  The identity hash owns the run root, so two identities
cannot share a mutable artifact namespace.  Attempt directories are never reused.  A failed or
crashed attempt may remain on disk, but it is invisible until a manifest is atomically published.

The supported concurrency model is one coordinating writer per run root.  After the first
mutation this store holds an advisory per-run writer lock; readers may validate committed
generations while that writer is active.  It does not provide hostile path-swap hardening or a
distributed/multi-run coordination protocol.  Load-time identity, manifest, and artifact hashes
are the guard against accidental corruption.

Only regular files with mode 0644 are accepted.  The implementation uses same-directory temp
files, file and directory fsyncs, and ``os.replace`` for publication.  ``fault_hook`` is an
optional test/integration seam; production callers leave it unset.  It receives operation names
``write``, ``fsync``, ``replace``, ``link``, and ``manifest_publication`` before the corresponding
operation and may raise to model a write, crash, or storage failure.
"""

from __future__ import annotations

from dataclasses import dataclass
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import tempfile
from typing import Any, Callable, Mapping
import uuid

try:
    import fcntl
except ImportError:  # pragma: no cover - supported targets are POSIX macOS/Linux
    fcntl = None  # type: ignore[assignment]

try:
    from .validator import canonical_sha256
except ImportError:  # pragma: no cover - direct air-gapped invocation
    from validator import canonical_sha256  # type: ignore


FILE_MODE = 0o644
DIRECTORY_MODE = 0o755
SCHEMA_VERSION = 1
_IDENTITY_FILENAME = "identity.json"
_IDENTITY_TEMP_PREFIX = f".{_IDENTITY_FILENAME}."
_IDENTITY_TEMP_SUFFIX = ".tmp"

FaultHook = Callable[[str, Path], None]


class ArtifactStoreFailure(ValueError):
    """Stable failure raised by the artifact store.

    ``kind`` is intentionally short and machine-readable.  The coordinator can turn it into a
    typed cell failure without depending on OS-specific exception text.
    """

    def __init__(self, kind: str, message: str):
        self.kind = kind
        self.reason = kind
        self.failure_type = "artifact_store_failure"
        self.message = message
        super().__init__(f"{kind}: {message}")


@dataclass(frozen=True)
class ArtifactEntry:
    """One validated file named by a committed manifest."""

    path: str
    size: int
    sha256: str
    mode: int = FILE_MODE

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "size": self.size,
            "sha256": self.sha256,
            "mode": self.mode,
        }


@dataclass(frozen=True)
class CommittedManifest:
    """A validated committed cell manifest.

    The mapping returned by :meth:`ArtifactStore.load` is usually more convenient for report
    code, but this typed view is useful to callers that want the attempt id or artifact entries
    without re-parsing them.
    """

    schema_version: int
    identity_sha256: str
    cell_id: str
    cell_sha256: str
    attempt_id: str
    artifacts: tuple[ArtifactEntry, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "identity_sha256": self.identity_sha256,
            "cell_id": self.cell_id,
            "cell_sha256": self.cell_sha256,
            "attempt_id": self.attempt_id,
            "artifacts": [entry.as_dict() for entry in self.artifacts],
        }


def _fail(kind: str, message: str) -> None:
    raise ArtifactStoreFailure(kind, message)


def _reject_nonfinite(value: Any, path: str = "$", seen: set[int] | None = None) -> None:
    """Reject values outside the plain JSON subset used by run identity and manifests."""

    value_type = type(value)
    if value is None or value_type in (bool, int):
        return
    if value_type is str:
        if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
            _fail("text_encoding", f"{path} contains an unpaired UTF-16 surrogate")
        return
    if value_type is float:
        if not math.isfinite(value):
            _fail("json_value", f"{path} contains a nonfinite number")
        return
    if value_type is dict:
        if seen is None:
            seen = set()
        marker = id(value)
        if marker in seen:
            _fail("json_value", f"{path} contains a cycle")
        seen.add(marker)
        for key, child in value.items():
            if type(key) is not str:
                _fail("json_value", f"{path} contains a non-string object key")
            _reject_nonfinite(child, f"{path}.{key}", seen)
        seen.remove(marker)
        return
    if value_type is list:
        if seen is None:
            seen = set()
        marker = id(value)
        if marker in seen:
            _fail("json_value", f"{path} contains a cycle")
        seen.add(marker)
        for index, child in enumerate(value):
            _reject_nonfinite(child, f"{path}[{index}]", seen)
        seen.remove(marker)
        return
    _fail("json_value", f"{path} contains a non-JSON value")


def _identity_copy(identity: Mapping[str, Any]) -> dict[str, Any]:
    """Return a deep, plain JSON identity and reject mutable/encoding surprises."""

    if not isinstance(identity, Mapping):
        _fail("identity_shape", "identity must be a mapping")
    # A top-level MappingProxy is useful to callers, but a shallow dict() copy would leave nested
    # lists/dicts aliased to caller-owned state.  Canonical JSON round-tripping gives the store a
    # detached tree and simultaneously proves every string can be encoded as UTF-8.
    copied = dict(identity)
    _reject_nonfinite(copied, "identity")
    try:
        encoded = json.dumps(
            copied,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        detached = json.loads(encoded.decode("utf-8"))
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ArtifactStoreFailure("identity_encoding", f"identity is not valid UTF-8 JSON: {exc}") from exc
    if type(detached) is not dict:  # pragma: no cover - top-level Mapping was copied above
        _fail("identity_shape", "identity must round-trip as a JSON object")
    _reject_nonfinite(detached, "identity")
    return detached


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _cell_sha256(cell_id: str) -> str:
    return _sha256_bytes(cell_id.encode("utf-8"))


def _safe_identifier(value: Any, path: str) -> str:
    if type(value) is not str or not value or "\x00" in value:
        _fail("path_safety", f"{path} must be a non-empty string without NUL")
    if any(0xD800 <= ord(character) <= 0xDFFF for character in value):
        _fail("text_encoding", f"{path} contains an unpaired UTF-16 surrogate")
    try:
        value.encode("utf-8")
    except UnicodeError as exc:  # defensive: surrogate check above should catch this
        raise ArtifactStoreFailure("text_encoding", f"{path} is not valid UTF-8: {exc}") from exc
    return value


def _safe_relative_path(value: Any, path: str = "artifact.path") -> str:
    """Validate a portable relative path without resolving symlinks."""

    name = _safe_identifier(value, path)
    # Artifact names are serialized as POSIX paths even on macOS.  Rejecting backslashes avoids
    # a name being interpreted as a separator if a future consumer runs on Windows.
    if "\\" in name:
        _fail("path_safety", f"{path} must use '/' separators")
    if name.startswith("/"):
        _fail("path_safety", f"{path} must be relative")
    parts = name.split("/")
    if any(part in ("", ".", "..") for part in parts):
        _fail("path_safety", f"{path} contains an unsafe component")
    if any("\x00" in part for part in parts):
        _fail("path_safety", f"{path} contains NUL")
    return "/".join(parts)


def _absolute(path: Path) -> Path:
    # Keep the lexical path intact.  Resolving here would silently approve a caller path such as
    # ``link/nested`` when ``link`` is a symlink.  The caller-facing constructor checks every
    # existing component before this helper is used; internal storage paths are checked with
    # lstat before every open/replace as well.
    return Path(os.path.abspath(os.fspath(path)))


def _lexical_absolute(path: Path) -> Path:
    """Make a path absolute without collapsing ``.`` or ``..`` components."""

    candidate = Path(path)
    return candidate if candidate.is_absolute() else Path.cwd() / candidate


def _reject_lexical_symlinks(path: Path) -> None:
    """Reject every existing symlink component in a caller-supplied lexical path."""

    lexical = _lexical_absolute(path)
    current = Path(lexical.anchor or os.sep)
    for component in lexical.parts[1:]:
        if component in ("", "."):
            continue
        if component == "..":
            current = current.parent
            continue
        current = current / component
        try:
            info = current.lstat()
        except FileNotFoundError:
            # Keep walking the lexical stack: a later ``..`` can return to an existing ancestor
            # whose subsequent component still needs symlink inspection.
            continue
        except OSError as exc:
            raise ArtifactStoreFailure("directory", f"cannot inspect caller path {current}: {exc}") from exc
        if stat.S_ISLNK(info.st_mode):
            _fail("symlink_path", f"caller path contains a symlink component: {current}")


def _ensure_directory(path: Path) -> None:
    """Create a directory chain while refusing symlink or non-directory components."""

    absolute = _absolute(path)
    current = Path(absolute.anchor or os.sep)
    for component in absolute.parts[1:]:
        current = current / component
        created = False
        try:
            info = current.lstat()
        except FileNotFoundError:
            try:
                current.mkdir()
                created = True
            except FileExistsError:
                # A concurrent creator is okay only if it produced a real directory.
                pass
            except OSError as exc:
                raise ArtifactStoreFailure("directory", f"cannot create {current}: {exc}") from exc
            try:
                info = current.lstat()
            except OSError as exc:
                raise ArtifactStoreFailure("directory", f"cannot inspect {current}: {exc}") from exc
        except OSError as exc:
            raise ArtifactStoreFailure("directory", f"cannot inspect {current}: {exc}") from exc
        if stat.S_ISLNK(info.st_mode):
            _fail("symlink_path", f"directory component is a symlink: {current}")
        if not stat.S_ISDIR(info.st_mode):
            _fail("directory", f"path component is not a directory: {current}")
        # Do not widen permissions on a caller-owned directory that already existed.  Newly
        # created directories are normalized after creation; directory mode is not part of the
        # artifact contract, but this keeps the tree usable under a restrictive umask.
        if created:
            try:
                os.chmod(current, DIRECTORY_MODE)
            except OSError as exc:
                raise ArtifactStoreFailure("directory", f"cannot set directory mode {current}: {exc}") from exc


def _check_directory(path: Path, *, kind: str = "directory") -> None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        _fail("missing_directory", f"{kind} does not exist: {path}")
    except OSError as exc:
        raise ArtifactStoreFailure(kind, f"cannot inspect {path}: {exc}") from exc
    if stat.S_ISLNK(info.st_mode):
        _fail("symlink_path", f"{kind} is a symlink: {path}")
    if not stat.S_ISDIR(info.st_mode):
        _fail("directory", f"{kind} is not a directory: {path}")


def _check_regular(
    path: Path,
    *,
    missing_ok: bool,
    kind: str = "artifact",
    require_mode: bool = True,
) -> None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        if missing_ok:
            return
        _fail("missing_artifact", f"{kind} does not exist: {path}")
    except OSError as exc:
        raise ArtifactStoreFailure(kind, f"cannot inspect {path}: {exc}") from exc
    if stat.S_ISLNK(info.st_mode):
        _fail("symlink_path", f"{kind} is a symlink: {path}")
    if not stat.S_ISREG(info.st_mode):
        _fail("nonregular_artifact", f"{kind} is not a regular file: {path}")
    if require_mode and stat.S_IMODE(info.st_mode) != FILE_MODE:
        _fail("artifact_mode", f"{kind} mode must be exactly 0644: {path}")


def _check_path_components(
    root: Path,
    relative: str,
    *,
    create_parents: bool,
    created_directories: list[Path] | None = None,
) -> Path:
    """Return ``root / relative`` after checking every existing component for symlinks."""

    parts = relative.split("/")
    current = root
    _check_directory(root, kind="attempt directory")
    for part in parts[:-1]:
        current = current / part
        created = False
        try:
            info = current.lstat()
        except FileNotFoundError:
            if not create_parents:
                _fail("missing_artifact", f"artifact parent does not exist: {current}")
            created = False
            try:
                current.mkdir()
                created = True
            except FileExistsError:
                pass
            except OSError as exc:
                raise ArtifactStoreFailure("directory", f"cannot create {current}: {exc}") from exc
            try:
                info = current.lstat()
            except OSError as exc:
                raise ArtifactStoreFailure("directory", f"cannot inspect {current}: {exc}") from exc
        except OSError as exc:
            raise ArtifactStoreFailure("directory", f"cannot inspect {current}: {exc}") from exc
        if stat.S_ISLNK(info.st_mode):
            _fail("symlink_path", f"artifact parent is a symlink: {current}")
        if not stat.S_ISDIR(info.st_mode):
            _fail("directory", f"artifact parent is not a directory: {current}")
        if created:
            if created_directories is not None:
                created_directories.append(current)
            try:
                os.chmod(current, DIRECTORY_MODE)
            except OSError as exc:
                raise ArtifactStoreFailure("directory", f"cannot set directory mode {current}: {exc}") from exc
    final = current / parts[-1]
    return final


def _remove_temporary(path: Path | None) -> None:
    if path is None:
        return
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        # The temporary name is private to this operation.  A cleanup failure must not hide the
        # original write/publication error, and a symlink unlink cannot affect its target.
        pass


def _strict_json_load(path: Path, payload: bytes) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                _fail("invalid_json", f"duplicate key {key!r} in {path}")
            result[key] = value
        return result

    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=reject_duplicates,
                           parse_constant=lambda constant: _fail("invalid_json", f"invalid constant {constant} in {path}"))
    except ArtifactStoreFailure:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactStoreFailure("invalid_json", f"cannot parse {path}: {exc}") from exc
    return value


def _read_regular(path: Path, *, kind: str) -> bytes:
    _check_regular(path, missing_ok=False, kind=kind)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = -1
    try:
        descriptor = os.open(os.fspath(path), flags)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    except ArtifactStoreFailure:
        raise
    except OSError as exc:
        raise ArtifactStoreFailure("read", f"cannot read {kind} {path}: {exc}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _invoke(hook: FaultHook | None, operation: str, path: Path) -> None:
    if hook is not None:
        hook(operation, path)


def _write_all(descriptor: int, payload: bytes, path: Path, hook: FaultHook | None) -> None:
    offset = 0
    while offset < len(payload):
        _invoke(hook, "write", path)
        try:
            written = os.write(descriptor, payload[offset:])
        except OSError as exc:
            raise ArtifactStoreFailure("write", f"cannot write {path}: {exc}") from exc
        if written <= 0:
            _fail("write", f"short write to {path}")
        offset += written


def _fsync(descriptor: int, path: Path, hook: FaultHook | None) -> None:
    _invoke(hook, "fsync", path)
    try:
        os.fsync(descriptor)
    except OSError as exc:
        raise ArtifactStoreFailure("fsync", f"cannot fsync {path}: {exc}") from exc


def _fsync_directory(directory: Path, hook: FaultHook | None) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = -1
    try:
        descriptor = os.open(os.fspath(directory), flags)
        _fsync(descriptor, directory, hook)
    except ArtifactStoreFailure:
        raise
    except OSError as exc:
        raise ArtifactStoreFailure("fsync", f"cannot open/fsync directory {directory}: {exc}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _atomic_file(
    path: Path,
    payload: bytes,
    *,
    hook: FaultHook | None,
    replace_existing: bool,
    publication_operation: str | None = None,
    source_path: Path | None = None,
) -> None:
    """Write one complete 0644 file and publish it in its parent directory."""

    _check_directory(path.parent, kind="file parent")
    _check_regular(path, missing_ok=True, kind="destination")
    descriptor = -1
    source_descriptor = -1
    temporary: Path | None = None
    try:
        try:
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{path.name}.", suffix=".tmp", dir=os.fspath(path.parent)
            )
        except OSError as exc:
            raise ArtifactStoreFailure("write", f"cannot create temporary file for {path}: {exc}") from exc
        temporary = Path(temporary_name)
        try:
            os.fchmod(descriptor, FILE_MODE)
        except OSError as exc:
            raise ArtifactStoreFailure("write", f"cannot set temporary mode for {path}: {exc}") from exc
        if source_path is None:
            _write_all(descriptor, payload, temporary, hook)
        else:
            # Source permissions do not become artifact permissions; the staged temporary is
            # always fchmod'd to 0644 above.  O_NOFOLLOW protects the source itself while the
            # caller's source remains open.
            _check_regular(source_path, missing_ok=False, kind="source file", require_mode=False)
            source_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            try:
                source_descriptor = os.open(os.fspath(source_path), source_flags)
            except OSError as exc:
                raise ArtifactStoreFailure("read", f"cannot open source file {source_path}: {exc}") from exc
            while True:
                try:
                    chunk = os.read(source_descriptor, 1024 * 1024)
                except OSError as exc:
                    raise ArtifactStoreFailure("read", f"cannot read source file {source_path}: {exc}") from exc
                if not chunk:
                    break
                _write_all(descriptor, chunk, temporary, hook)
        _fsync(descriptor, temporary, hook)
        if source_descriptor >= 0:
            try:
                os.close(source_descriptor)
            except OSError as exc:
                raise ArtifactStoreFailure("read", f"cannot close source file {source_path}: {exc}") from exc
            source_descriptor = -1
        try:
            os.close(descriptor)
        except OSError as exc:
            raise ArtifactStoreFailure("write", f"cannot close temporary file for {path}: {exc}") from exc
        descriptor = -1

        if replace_existing:
            if publication_operation is not None:
                _invoke(hook, publication_operation, path)
            _invoke(hook, "replace", path)
            try:
                os.replace(os.fspath(temporary), os.fspath(path))
            except OSError as exc:
                raise ArtifactStoreFailure("replace", f"cannot publish {path}: {exc}") from exc
        else:
            _invoke(hook, "link", path)
            try:
                os.link(os.fspath(temporary), os.fspath(path))
            except FileExistsError as exc:
                raise ArtifactStoreFailure("already_exists", f"destination appeared during publication: {path}") from exc
            except OSError as exc:
                raise ArtifactStoreFailure("link", f"cannot publish {path}: {exc}") from exc
        _remove_temporary(temporary)
        temporary = None
        _fsync_directory(path.parent, hook)
    except ArtifactStoreFailure:
        raise
    except OSError as exc:
        raise ArtifactStoreFailure("io", f"artifact publication failed for {path}: {exc}") from exc
    finally:
        if source_descriptor >= 0:
            try:
                os.close(source_descriptor)
            except OSError:
                pass
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass
        _remove_temporary(temporary)


def _hash_file(path: Path) -> tuple[int, str]:
    _check_regular(path, missing_ok=False, kind="artifact")
    descriptor = -1
    digest = hashlib.sha256()
    size = 0
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(os.fspath(path), flags)
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
    except OSError as exc:
        raise ArtifactStoreFailure("read", f"cannot hash artifact {path}: {exc}") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    # Re-check metadata after reading.  A mode or file replacement race must not be silently
    # included in a manifest.  The hash itself is still best-effort against concurrent writes;
    # integration should not mutate an active attempt while committing it.
    _check_regular(path, missing_ok=False, kind="artifact")
    return size, digest.hexdigest()


def _walk_artifacts(root: Path) -> list[tuple[str, Path]]:
    """Walk an attempt without following symlinks and return portable relative names."""

    _check_directory(root, kind="attempt directory")
    result: list[tuple[str, Path]] = []

    def visit(directory: Path, prefix: str) -> None:
        try:
            entries = sorted(directory.iterdir(), key=lambda child: child.name)
        except OSError as exc:
            raise ArtifactStoreFailure("read", f"cannot list attempt directory {directory}: {exc}") from exc
        for entry in entries:
            relative = f"{prefix}/{entry.name}" if prefix else entry.name
            _safe_relative_path(relative)
            try:
                info = entry.lstat()
            except OSError as exc:
                raise ArtifactStoreFailure("read", f"cannot inspect attempt entry {entry}: {exc}") from exc
            if stat.S_ISLNK(info.st_mode):
                _fail("symlink_path", f"attempt contains a symlink: {entry}")
            if stat.S_ISDIR(info.st_mode):
                visit(entry, relative)
            elif stat.S_ISREG(info.st_mode):
                if stat.S_IMODE(info.st_mode) != FILE_MODE:
                    _fail("artifact_mode", f"artifact mode must be exactly 0644: {entry}")
                result.append((relative, entry))
            else:
                _fail("nonregular_artifact", f"attempt contains a non-regular entry: {entry}")

    visit(root, "")
    return result


def _manifest_from_dict(value: Any, *, expected_identity: str, expected_cell: str | None = None) -> CommittedManifest:
    if type(value) is not dict:
        _fail("manifest_shape", "committed manifest must be a plain JSON object")
    expected_keys = {
        "schema_version", "identity_sha256", "cell_id", "cell_sha256", "attempt_id", "artifacts"
    }
    if set(value) != expected_keys:
        _fail("manifest_shape", "committed manifest has unsupported or missing fields")
    schema_version = value["schema_version"]
    if type(schema_version) is not int or schema_version != SCHEMA_VERSION:
        _fail("schema_version", "committed manifest schema_version must be integer 1")
    identity_sha = value["identity_sha256"]
    if type(identity_sha) is not str or len(identity_sha) != 64 or identity_sha != identity_sha.lower():
        _fail("identity_hash", "committed manifest identity_sha256 must be lowercase SHA256")
    try:
        int(identity_sha, 16)
    except ValueError as exc:
        raise ArtifactStoreFailure("identity_hash", "committed manifest identity_sha256 is not hexadecimal") from exc
    if identity_sha != expected_identity:
        _fail("identity_mismatch", "committed manifest belongs to a different run identity")
    cell_id = _safe_identifier(value["cell_id"], "manifest.cell_id")
    if expected_cell is not None and cell_id != expected_cell:
        _fail("cell_mismatch", "committed manifest cell_id does not match requested cell")
    cell_sha = value["cell_sha256"]
    if type(cell_sha) is not str or len(cell_sha) != 64 or cell_sha != cell_sha.lower() or cell_sha != _cell_sha256(cell_id):
        _fail("cell_hash", "committed manifest cell_sha256 does not match cell_id")
    attempt_id = _safe_identifier(value["attempt_id"], "manifest.attempt_id")
    try:
        parsed_attempt = uuid.UUID(attempt_id)
    except (ValueError, AttributeError) as exc:
        _fail("attempt_id", "committed manifest attempt_id must be a UUID")
    if str(parsed_attempt) != attempt_id.lower():
        _fail("attempt_id", "committed manifest attempt_id must use canonical UUID spelling")
    artifacts_value = value["artifacts"]
    if type(artifacts_value) is not list:
        _fail("manifest_shape", "committed manifest artifacts must be a list")
    entries: list[ArtifactEntry] = []
    previous = ""
    for index, raw in enumerate(artifacts_value):
        if type(raw) is not dict or set(raw) != {"path", "size", "sha256", "mode"}:
            _fail("artifact_shape", f"manifest.artifacts[{index}] has the wrong fields")
        relative = _safe_relative_path(raw["path"], f"manifest.artifacts[{index}].path")
        if relative <= previous:
            _fail("artifact_order", "manifest artifacts must be unique and sorted by path")
        previous = relative
        size = raw["size"]
        if type(size) is not int or size < 0:
            _fail("artifact_shape", f"manifest.artifacts[{index}].size must be a nonnegative integer")
        digest = raw["sha256"]
        if type(digest) is not str or len(digest) != 64 or digest != digest.lower():
            _fail("artifact_hash", f"manifest.artifacts[{index}].sha256 must be lowercase SHA256")
        try:
            int(digest, 16)
        except ValueError as exc:
            raise ArtifactStoreFailure("artifact_hash", f"manifest.artifacts[{index}].sha256 is not hexadecimal") from exc
        mode = raw["mode"]
        if type(mode) is not int or mode != FILE_MODE:
            _fail("artifact_mode", f"manifest.artifacts[{index}].mode must be 0644")
        entries.append(ArtifactEntry(relative, size, digest, mode))
    return CommittedManifest(SCHEMA_VERSION, identity_sha, cell_id, cell_sha, attempt_id, tuple(entries))


_ARTIFACT_REF_KEYS = {
    "schema_version", "identity_sha256", "cell_id", "cell_sha256", "attempt_id", "manifest_sha256"
}


def _manifest_sha256(manifest: CommittedManifest) -> str:
    """Hash the canonical manifest object, excluding the external reference by construction."""

    return canonical_sha256(manifest.as_dict())


def _artifact_ref(manifest: CommittedManifest) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "identity_sha256": manifest.identity_sha256,
        "cell_id": manifest.cell_id,
        "cell_sha256": manifest.cell_sha256,
        "attempt_id": manifest.attempt_id,
        "manifest_sha256": _manifest_sha256(manifest),
    }


def _artifact_ref_from_dict(
    value: Any,
    *,
    expected_identity: str,
    expected_cell: str | None = None,
) -> dict[str, Any]:
    """Validate the small pointer/reference object without following it."""

    if type(value) is not dict or set(value) != _ARTIFACT_REF_KEYS:
        _fail("artifact_ref_shape", "artifact_ref must contain exactly its six reference fields")
    schema_version = value["schema_version"]
    if type(schema_version) is not int or schema_version != SCHEMA_VERSION:
        _fail("schema_version", "artifact_ref schema_version must be integer 1")
    identity_sha = value["identity_sha256"]
    if type(identity_sha) is not str or len(identity_sha) != 64 or identity_sha != identity_sha.lower():
        _fail("identity_hash", "artifact_ref identity_sha256 must be lowercase SHA256")
    try:
        int(identity_sha, 16)
    except ValueError as exc:
        raise ArtifactStoreFailure("identity_hash", "artifact_ref identity_sha256 is not hexadecimal") from exc
    if identity_sha != expected_identity:
        _fail("identity_mismatch", "artifact_ref belongs to a different run identity")
    cell_id = _safe_identifier(value["cell_id"], "artifact_ref.cell_id")
    if expected_cell is not None and cell_id != expected_cell:
        _fail("cell_mismatch", "artifact_ref cell_id does not match requested cell")
    cell_sha = value["cell_sha256"]
    if type(cell_sha) is not str or len(cell_sha) != 64 or cell_sha != cell_sha.lower() or cell_sha != _cell_sha256(cell_id):
        _fail("cell_hash", "artifact_ref cell_sha256 does not match cell_id")
    attempt_id = _safe_identifier(value["attempt_id"], "artifact_ref.attempt_id")
    try:
        parsed_attempt = uuid.UUID(attempt_id)
    except (ValueError, AttributeError) as exc:
        raise ArtifactStoreFailure("attempt_id", "artifact_ref attempt_id must be a UUID") from exc
    if str(parsed_attempt) != attempt_id.lower():
        _fail("attempt_id", "artifact_ref attempt_id must use canonical UUID spelling")
    manifest_sha = value["manifest_sha256"]
    if type(manifest_sha) is not str or len(manifest_sha) != 64 or manifest_sha != manifest_sha.lower():
        _fail("manifest_hash", "artifact_ref manifest_sha256 must be lowercase SHA256")
    try:
        int(manifest_sha, 16)
    except ValueError as exc:
        raise ArtifactStoreFailure("manifest_hash", "artifact_ref manifest_sha256 is not hexadecimal") from exc
    return {
        "schema_version": SCHEMA_VERSION,
        "identity_sha256": identity_sha,
        "cell_id": cell_id,
        "cell_sha256": cell_sha,
        "attempt_id": attempt_id,
        "manifest_sha256": manifest_sha,
    }


class ArtifactAttempt:
    """One unique, unpublished cell attempt."""

    def __init__(self, store: "ArtifactStore", cell_id: str, attempt_id: str, root: Path):
        self.store = store
        self.cell_id = cell_id
        self.attempt_id = attempt_id
        self.root = root
        self._closed = False
        self._artifact_ref: dict[str, Any] | None = None
        self._recovery_error: ArtifactStoreFailure | None = None

    def _check_open(self) -> None:
        if self._closed:
            _fail("attempt_closed", "attempt is closed and cannot be changed")
        self.store._check_open()
        _check_directory(self.root, kind="attempt directory")

    def _check_store_open(self) -> None:
        """Reject access after the owning store has released its writer lease."""

        self.store._check_open()

    @property
    def artifact_ref(self) -> dict[str, Any] | None:
        """Return the exact committed reference recovered for this attempt, if any."""

        return None if self._artifact_ref is None else dict(self._artifact_ref)

    def reconcile(self) -> dict[str, Any] | None:
        """Reconcile an interrupted commit against the durable current pointer.

        A directory fsync can fail after ``os.replace`` has made ``committed.json`` visible.  The
        attempt is closed before the original exception escapes, and this method deterministically
        returns the exact reference if that pointer names this attempt.  It returns ``None`` when
        the pointer still names an older generation (or no generation).
        """

        # Reconciliation is intentionally allowed for a closed attempt while its owning store
        # is still alive: commit() uses this path to recover a pointer published just before a
        # directory fsync failure.  Once the store closes, however, every old attempt becomes
        # inert, including an attempt that already has a recovered reference.
        self._check_store_open()
        if self._artifact_ref is not None:
            return dict(self._artifact_ref)
        try:
            current = self.store._load_current_ref(self.cell_id)
        except ArtifactStoreFailure as exc:
            self._recovery_error = exc
            raise
        if current is not None and current["attempt_id"] == self.attempt_id:
            self.store.load_ref(current)
            self._artifact_ref = current
        return None if self._artifact_ref is None else dict(self._artifact_ref)

    def stage_bytes(self, relative_path: str, payload: bytes | bytearray | memoryview) -> Path:
        """Atomically stage arbitrary bytes under this attempt and return the final path."""

        self._check_open()
        relative = _safe_relative_path(relative_path)
        if not isinstance(payload, (bytes, bytearray, memoryview)):
            _fail("write", "payload must be bytes-like")
        publishing = False
        try:
            final = _check_path_components(
                self.root,
                relative,
                create_parents=True,
            )
            _check_regular(final, missing_ok=True, kind="destination")
            publishing = True
            _atomic_file(final, bytes(payload), hook=self.store.fault_hook, replace_existing=True)
            # _atomic_file syncs the immediate parent.  Sync every artifact parent in bottom-up
            # order (including existing ancestors whose directory entry gained a new child),
            # then the attempt root, before any manifest can name this file.
            parent_parts = relative.split("/")[:-1]
            for depth in range(len(parent_parts), 0, -1):
                _fsync_directory(self.root.joinpath(*parent_parts[:depth]), self.store.fault_hook)
            _fsync_directory(self.root, self.store.fault_hook)
            return final
        except ArtifactStoreFailure:
            if publishing:
                self._closed = True
            raise

    def stage_file(self, relative_path: str, source: Path | str) -> Path:
        """Copy a regular source file into the attempt with mode 0644 and atomic replacement."""

        self._check_open()
        relative = _safe_relative_path(relative_path)
        source_path = Path(source)
        # The source may be private (0600) or otherwise use a caller-selected mode.  Only files
        # persisted inside the attempt are constrained to 0644.
        _check_regular(source_path, missing_ok=False, kind="source file", require_mode=False)
        publishing = False
        try:
            final = _check_path_components(
                self.root,
                relative,
                create_parents=True,
            )
            _check_regular(final, missing_ok=True, kind="destination")
            publishing = True
            _atomic_file(
                final,
                b"",
                hook=self.store.fault_hook,
                replace_existing=True,
                source_path=source_path,
            )
            parent_parts = relative.split("/")[:-1]
            for depth in range(len(parent_parts), 0, -1):
                _fsync_directory(self.root.joinpath(*parent_parts[:depth]), self.store.fault_hook)
            _fsync_directory(self.root, self.store.fault_hook)
            return final
        except ArtifactStoreFailure:
            if publishing:
                self._closed = True
            raise

    def validate(self) -> tuple[ArtifactEntry, ...]:
        """Validate every attempt file and return sorted size/hash records."""

        self._check_open()
        entries: list[ArtifactEntry] = []
        for relative, path in _walk_artifacts(self.root):
            size, digest = _hash_file(path)
            entries.append(ArtifactEntry(relative, size, digest))
        return tuple(entries)

    def commit(self) -> dict[str, Any]:
        """Validate this attempt and publish an immutable manifest plus current pointer.

        The returned value is an exact ``artifact_ref``.  It remains valid after another attempt
        replaces the cell's current pointer.  Any failure closes this attempt; callers can use
        :meth:`reconcile` to determine whether a post-replace failure actually committed it.
        """

        self._check_open()
        try:
            entries = self.validate()
            manifest = CommittedManifest(
                SCHEMA_VERSION,
                self.store.identity_sha256,
                self.cell_id,
                _cell_sha256(self.cell_id),
                self.attempt_id,
                entries,
            )
            ref = _artifact_ref(manifest)
            manifest_payload = (
                json.dumps(manifest.as_dict(), ensure_ascii=False, sort_keys=True, indent=2) + "\n"
            ).encode("utf-8")
            self.store._publish_immutable_manifest(manifest, manifest_payload)
            pointer_payload = (json.dumps(ref, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
            _atomic_file(
                self.store._manifest_path(self.cell_id),
                pointer_payload,
                hook=self.store.fault_hook,
                replace_existing=True,
                publication_operation="manifest_publication",
            )
            self._artifact_ref = ref
            self._closed = True
            return dict(ref)
        except ArtifactStoreFailure as exc:
            # A failure may have occurred after os.replace installed the pointer.  Closing first
            # prevents a caller from mutating an attempt whose generation may already be current.
            self._closed = True
            try:
                recovered = self.reconcile()
            except ArtifactStoreFailure as recovery_error:
                self._recovery_error = recovery_error
                recovered = None
            if recovered is not None:
                setattr(exc, "artifact_ref", dict(recovered))
            raise


class ArtifactStore:
    """Identity-owned transactional artifact store.

    ``parent`` is a namespace containing one run root per identity.  Constructing the store is
    idempotent for the same identity and refuses a root whose persisted identity differs or is
    absent.  The store supports one coordinating writer per run and acquires an advisory per-run
    writer lock after the first mutation; it does not claim hostile path-swap or broader
    multi-writer safety.  ``fault_hook`` exists for deterministic fault-injection
    tests; it is not needed by normal callers.
    """

    def __init__(
        self,
        parent: Path | str,
        identity: Mapping[str, Any],
        *,
        fault_hook: FaultHook | None = None,
    ) -> None:
        raw_parent = Path(parent)
        _reject_lexical_symlinks(raw_parent)
        try:
            raw_info = raw_parent.lstat()
        except FileNotFoundError:
            raw_info = None
        except OSError as exc:
            raise ArtifactStoreFailure("directory", f"cannot inspect artifact parent {raw_parent}: {exc}") from exc
        if raw_info is not None and stat.S_ISLNK(raw_info.st_mode):
            _fail("symlink_path", f"artifact parent is a symlink: {raw_parent}")
        self.parent = _absolute(raw_parent)
        self._identity = _identity_copy(identity)
        try:
            self.identity_sha256 = canonical_sha256(self._identity)
        except (TypeError, ValueError, UnicodeError) as exc:
            raise ArtifactStoreFailure("identity_hash", f"cannot hash identity: {exc}") from exc
        self.fault_hook = fault_hook
        _ensure_directory(self.parent)
        self.run_root = self.parent / self.identity_sha256
        self._writer_lock_fd = -1
        self._closed = False
        self._create_or_check_run_root()
        self.cells_root = self.run_root / "cells"
        _ensure_directory(self.cells_root)
        # ``_ensure_directory`` deliberately only checks/creates components.  Persist the
        # containing run-root entry after the initial cells directory is created so a crash
        # cannot leave durable children whose parent entry was never committed.
        _fsync_directory(self.run_root, self.fault_hook)

    @property
    def identity(self) -> dict[str, Any]:
        """Return a detached identity copy; callers cannot mutate store identity in place."""

        return _identity_copy(self._identity)

    def _create_or_check_run_root(self) -> None:
        created = False
        try:
            info = self.run_root.lstat()
        except FileNotFoundError:
            try:
                self.run_root.mkdir()
                created = True
                info = self.run_root.lstat()
            except FileExistsError:
                # Another same-identity constructor won the directory creation race.  Inspect
                # the resulting entry below rather than treating the benign race as a failure.
                try:
                    info = self.run_root.lstat()
                except OSError as exc:
                    raise ArtifactStoreFailure("directory", f"cannot inspect identity run root {self.run_root}: {exc}") from exc
        except OSError as exc:
            raise ArtifactStoreFailure("directory", f"cannot inspect identity run root {self.run_root}: {exc}") from exc
        if stat.S_ISLNK(info.st_mode):
            _fail("symlink_path", f"identity run root is a symlink: {self.run_root}")
        if not stat.S_ISDIR(info.st_mode):
            _fail("directory", f"identity run root is not a directory: {self.run_root}")
        if created:
            try:
                os.chmod(self.run_root, DIRECTORY_MODE)
            except OSError as exc:
                raise ArtifactStoreFailure("directory", f"cannot create identity run root {self.run_root}: {exc}") from exc
            # The identity root itself was created with mkdir; fsyncing the child later does not
            # make the new entry durable in its containing namespace.
            _fsync_directory(self.parent, self.fault_hook)
        identity_path = self.run_root / _IDENTITY_FILENAME
        payload = (
            json.dumps(
                {"schema_version": SCHEMA_VERSION, "identity": self._identity, "identity_sha256": self.identity_sha256},
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
            + "\n"
        ).encode("utf-8")
        try:
            identity_path.lstat()
        except FileNotFoundError:
            identity_exists = False
        except OSError as exc:
            raise ArtifactStoreFailure("identity", f"cannot inspect identity file {identity_path}: {exc}") from exc
        else:
            identity_exists = True
        if identity_exists:
            self._check_identity_file(identity_path)
        else:
            try:
                existing_entries = list(self.run_root.iterdir())
            except OSError as exc:
                raise ArtifactStoreFailure("directory", f"cannot inspect identity run root: {exc}") from exc
            unknown_entries: list[Path] = []
            for entry in existing_entries:
                if entry.name == _IDENTITY_FILENAME:
                    continue
                if entry.name.startswith(_IDENTITY_TEMP_PREFIX) and entry.name.endswith(_IDENTITY_TEMP_SUFFIX):
                    # This exact pattern is the private tempfile naming convention used by
                    # ``_atomic_file(identity.json, ...)``.  Leave it in place: another
                    # constructor may still be writing it.  It is never loaded as identity data,
                    # and the next successful publication makes the run root unambiguous.
                    try:
                        info = entry.lstat()
                    except OSError as exc:
                        raise ArtifactStoreFailure("identity", f"cannot inspect identity temporary {entry}: {exc}") from exc
                    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
                        _fail("identity_missing", f"identity temporary is not a regular file: {entry}")
                    continue
                unknown_entries.append(entry)
            if unknown_entries:
                _fail("identity_missing", f"identity run root has content but no identity file: {self.run_root}")
            try:
                _atomic_file(
                    identity_path,
                    payload,
                    hook=self.fault_hook,
                    replace_existing=False,
                )
            except ArtifactStoreFailure as exc:
                if exc.kind != "already_exists":
                    raise
            self._check_identity_file(identity_path)

    def _check_identity_file(self, path: Path) -> None:
        _check_regular(path, missing_ok=False, kind="identity file")
        value = _strict_json_load(path, _read_regular(path, kind="identity file"))
        if type(value) is not dict or set(value) != {"schema_version", "identity", "identity_sha256"}:
            _fail("identity_shape", "identity file has unsupported or missing fields")
        if value["schema_version"] != SCHEMA_VERSION or type(value["schema_version"]) is not int:
            _fail("schema_version", "identity file schema_version must be integer 1")
        persisted = _identity_copy(value["identity"])
        persisted_hash = value["identity_sha256"]
        if type(persisted_hash) is not str or persisted_hash != self.identity_sha256:
            _fail("identity_mismatch", "identity file belongs to a different identity")
        if canonical_sha256(persisted) != persisted_hash or persisted != self._identity:
            _fail("identity_mismatch", "identity file contents do not match this run identity")

    def _acquire_writer_lock(self) -> None:
        """Acquire the run's advisory single-coordinator writer lock on first mutation."""

        self._check_open()
        if self._writer_lock_fd >= 0:
            return
        if fcntl is None:  # pragma: no cover - all supported hosts provide fcntl
            _fail("writer_lock", "single-writer enforcement requires POSIX advisory locking")
        lock_path = self.run_root / "writer.lock"
        _check_regular(lock_path, missing_ok=True, kind="writer lock")
        descriptor = -1
        try:
            descriptor = os.open(
                os.fspath(lock_path),
                os.O_WRONLY | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
                FILE_MODE,
            )
            os.fchmod(descriptor, FILE_MODE)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                if exc.errno in (errno.EACCES, errno.EAGAIN):
                    raise ArtifactStoreFailure("writer_busy", f"another coordinator owns run writer lock: {lock_path}") from exc
                raise ArtifactStoreFailure("writer_lock", f"cannot acquire writer lock {lock_path}: {exc}") from exc
            self._writer_lock_fd = descriptor
            descriptor = -1
        except ArtifactStoreFailure:
            raise
        except OSError as exc:
            raise ArtifactStoreFailure("writer_lock", f"cannot open writer lock {lock_path}: {exc}") from exc
        finally:
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except OSError:
                    pass

    def close(self) -> None:
        """Close the store and invalidate all attempts it previously issued.

        Closing is a store-lifetime boundary, not merely a file-descriptor cleanup.  Attempts
        retain a back-reference to this object and check it before every public operation, so a
        stale attempt cannot write after this store releases the run lock and another coordinator
        acquires it.
        """

        self._closed = True
        descriptor, self._writer_lock_fd = self._writer_lock_fd, -1
        if descriptor < 0:
            return
        try:
            if fcntl is not None:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            try:
                os.close(descriptor)
            except OSError:
                pass

    def _check_open(self) -> None:
        if self._closed:
            _fail("store_closed", "artifact store is closed and cannot be used")

    def __enter__(self) -> "ArtifactStore":
        return self

    def __exit__(self, _exc_type: Any, _exc_value: Any, _traceback: Any) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - best-effort process teardown cleanup
        try:
            self.close()
        except Exception:
            pass

    def _cell_root(self, cell_id: str, *, create: bool) -> Path:
        cell_id = _safe_identifier(cell_id, "cell_id")
        root = self.cells_root / _cell_sha256(cell_id)
        if create:
            _ensure_directory(root)
            _ensure_directory(root / "attempts")
            _ensure_directory(root / "manifests")
        else:
            _check_directory(root, kind="cell directory")
            _check_directory(root / "attempts", kind="attempts directory")
            _check_directory(root / "manifests", kind="manifests directory")
        return root

    def _manifest_path(self, cell_id: str) -> Path:
        """Return the replaceable latest-pointer path (kept as a compatibility name)."""

        return self._cell_root(cell_id, create=True) / "committed.json"

    def _immutable_manifest_path(self, cell_id: str, attempt_id: str, *, create: bool) -> Path:
        cell_root = self._cell_root(cell_id, create=create)
        _safe_identifier(attempt_id, "attempt_id")
        return cell_root / "manifests" / f"{attempt_id}.json"

    def _publish_immutable_manifest(self, manifest: CommittedManifest, payload: bytes) -> None:
        """Publish one attempt manifest with no-clobber semantics.

        A retry after an ambiguous pointer publication may re-enter this helper only through a
        fresh attempt object in normal operation.  The no-clobber check nevertheless makes a
        duplicate call idempotent when the same attempt's manifest already reached disk.
        """

        path = self._immutable_manifest_path(manifest.cell_id, manifest.attempt_id, create=True)
        _check_regular(path, missing_ok=True, kind="immutable manifest")
        if path.exists() or path.is_symlink():
            existing = _strict_json_load(path, _read_regular(path, kind="immutable manifest"))
            parsed = _manifest_from_dict(existing, expected_identity=self.identity_sha256, expected_cell=manifest.cell_id)
            if parsed.as_dict() != manifest.as_dict() or _manifest_sha256(parsed) != _manifest_sha256(manifest):
                _fail("manifest_collision", f"immutable manifest already exists with different content: {path}")
            return
        try:
            _atomic_file(
                path,
                payload,
                hook=self.fault_hook,
                replace_existing=False,
            )
        except ArtifactStoreFailure as exc:
            if exc.kind != "already_exists":
                raise
            # A concurrent publisher won the no-clobber race.  Validate that it published this
            # exact generation rather than accepting an arbitrary same-name file.
            existing = _strict_json_load(path, _read_regular(path, kind="immutable manifest"))
            parsed = _manifest_from_dict(existing, expected_identity=self.identity_sha256, expected_cell=manifest.cell_id)
            if parsed.as_dict() != manifest.as_dict() or _manifest_sha256(parsed) != _manifest_sha256(manifest):
                _fail("manifest_collision", f"immutable manifest race produced different content: {path}")

    def begin(self, cell_id: str) -> ArtifactAttempt:
        """Create a unique unpublished attempt for ``cell_id``."""

        self._check_open()
        cell_id = _safe_identifier(cell_id, "cell_id")
        self._acquire_writer_lock()
        try:
            cell_root = self._cell_root(cell_id, create=True)
            attempts_root = cell_root / "attempts"
            while True:
                attempt_id = str(uuid.uuid4())
                attempt_root = attempts_root / attempt_id
                try:
                    attempt_root.mkdir()
                    os.chmod(attempt_root, DIRECTORY_MODE)
                    # Persist the directory entries before any result can be published.  If a host
                    # crashes after this point, a later loader can either find the attempt named by
                    # the manifest or safely ignore this abandoned directory.
                    _fsync_directory(self.cells_root, self.fault_hook)
                    _fsync_directory(cell_root, self.fault_hook)
                    _fsync_directory(attempts_root, self.fault_hook)
                    break
                except FileExistsError:
                    continue
                except OSError as exc:
                    raise ArtifactStoreFailure("attempt", f"cannot create attempt directory {attempt_root}: {exc}") from exc
            return ArtifactAttempt(self, cell_id, attempt_id, attempt_root)
        except BaseException:
            # begin() is the first mutating operation and may fail after acquiring the lock while
            # creating/fsyncing the attempt tree.  The supported lifetime model is deliberately
            # simple: poison this store, release its lease, and require a fresh ArtifactStore.
            # This prevents a partially-created attempt from being reused and lets another
            # coordinator acquire the lock immediately.
            self.close()
            raise

    def _validate_committed_attempt(self, manifest: CommittedManifest) -> CommittedManifest:
        """Validate the exact attempt files named by an immutable manifest."""

        cell_root = self._cell_root(manifest.cell_id, create=False)
        attempt_root = cell_root / "attempts" / manifest.attempt_id
        _check_directory(attempt_root, kind="committed attempt directory")
        actual = _walk_artifacts(attempt_root)
        actual_by_path = {relative: path for relative, path in actual}
        expected_by_path = {entry.path: entry for entry in manifest.artifacts}
        if set(actual_by_path) != set(expected_by_path):
            _fail("artifact_set", "committed attempt files do not match manifest")
        for relative, entry in expected_by_path.items():
            path = actual_by_path[relative]
            size, digest = _hash_file(path)
            if size != entry.size or digest != entry.sha256:
                _fail("artifact_hash_mismatch", f"committed artifact does not match manifest: {relative}")
        return manifest

    def _load_ref(self, ref: Mapping[str, Any]) -> CommittedManifest:
        validated_ref = _artifact_ref_from_dict(dict(ref), expected_identity=self.identity_sha256)
        path = self._immutable_manifest_path(validated_ref["cell_id"], validated_ref["attempt_id"], create=False)
        _check_regular(path, missing_ok=False, kind="immutable manifest")
        value = _strict_json_load(path, _read_regular(path, kind="immutable manifest"))
        manifest = _manifest_from_dict(
            value,
            expected_identity=self.identity_sha256,
            expected_cell=validated_ref["cell_id"],
        )
        if manifest.attempt_id != validated_ref["attempt_id"] or manifest.cell_sha256 != validated_ref["cell_sha256"]:
            _fail("artifact_ref_mismatch", "immutable manifest does not match artifact_ref")
        if _manifest_sha256(manifest) != validated_ref["manifest_sha256"]:
            _fail("manifest_hash_mismatch", "immutable manifest does not match artifact_ref hash")
        return self._validate_committed_attempt(manifest)

    def load_ref(self, ref: Mapping[str, Any]) -> dict[str, Any]:
        """Load and validate the exact immutable generation named by ``ref``."""

        return self._load_ref(ref).as_dict()

    def _load_current_ref(self, cell_id: str) -> dict[str, Any] | None:
        cell_id = _safe_identifier(cell_id, "cell_id")
        candidate_root = self.cells_root / _cell_sha256(cell_id)
        try:
            candidate_root.lstat()
        except FileNotFoundError:
            # A cell that has never started is the same public result as a cell with only
            # abandoned attempts: no current pointer exists.
            return None
        except OSError as exc:
            raise ArtifactStoreFailure("directory", f"cannot inspect cell directory {candidate_root}: {exc}") from exc
        cell_root = self._cell_root(cell_id, create=False)
        path = cell_root / "committed.json"
        _check_regular(path, missing_ok=True, kind="committed pointer")
        try:
            path.lstat()
        except FileNotFoundError:
            return None
        value = _strict_json_load(path, _read_regular(path, kind="committed pointer"))
        ref = _artifact_ref_from_dict(value, expected_identity=self.identity_sha256, expected_cell=cell_id)
        # Validate the pointer target before returning it.  A current pointer to missing or
        # tampered evidence is a hard integrity failure, not a reason to fall back to an older
        # generation.
        self._load_ref(ref)
        return ref

    def _load_manifest(self, cell_id: str) -> CommittedManifest | None:
        current = self._load_current_ref(cell_id)
        return None if current is None else self._load_ref(current)

    def load(self, cell_id: str) -> dict[str, Any] | None:
        """Load the manifest named by the cell's current pointer, or ``None`` if absent."""

        manifest = self._load_manifest(cell_id)
        return None if manifest is None else manifest.as_dict()

    def current_ref(self, cell_id: str) -> dict[str, Any] | None:
        """Return a validated detached exact reference for the cell's current generation."""

        ref = self._load_current_ref(cell_id)
        return None if ref is None else dict(ref)

    def load_all(self) -> dict[str, dict[str, Any]]:
        """Load every committed cell; abandoned attempts are never considered."""

        _check_directory(self.cells_root, kind="cells directory")
        result: dict[str, dict[str, Any]] = {}
        try:
            cell_dirs = sorted(self.cells_root.iterdir(), key=lambda path: path.name)
        except OSError as exc:
            raise ArtifactStoreFailure("read", f"cannot list cells directory: {exc}") from exc
        for cell_root in cell_dirs:
            info = cell_root.lstat()
            if stat.S_ISLNK(info.st_mode):
                _fail("symlink_path", f"cell entry is a symlink: {cell_root}")
            if not stat.S_ISDIR(info.st_mode):
                _fail("directory", f"cell entry is not a directory: {cell_root}")
            manifest_path = cell_root / "committed.json"
            _check_regular(manifest_path, missing_ok=True, kind="committed manifest")
            if not manifest_path.exists():
                # A cell may have only abandoned attempts.  It is deliberately invisible.
                continue
            value = _strict_json_load(manifest_path, _read_regular(manifest_path, kind="committed pointer"))
            ref = _artifact_ref_from_dict(value, expected_identity=self.identity_sha256)
            if cell_root.name != ref["cell_sha256"]:
                _fail("cell_hash", f"cell directory does not match committed cell_id: {cell_root}")
            # Reuse the single exact-reference path, which verifies the immutable manifest and
            # every artifact hash/mode.
            loaded = self._load_ref(ref)
            if loaded is None:  # pragma: no cover - _load_ref is not optional
                _fail("missing_manifest", f"committed manifest disappeared: {manifest_path}")
            result[ref["cell_id"]] = loaded.as_dict()
        return result

    def artifact_path(self, ref: Mapping[str, Any], relative_path: str) -> Path:
        """Resolve one artifact from an exact ``artifact_ref`` generation.

        Public artifact access never accepts a free-standing manifest.  Loading the exact
        immutable generation first prevents a fabricated manifest from being used to read an
        uncommitted attempt.
        """

        value = self._load_ref(ref)
        relative = _safe_relative_path(relative_path)
        if relative not in {entry.path for entry in value.artifacts}:
            _fail("unknown_artifact", f"manifest does not name artifact {relative!r}")
        attempt_root = self._cell_root(value.cell_id, create=False) / "attempts" / value.attempt_id
        path = _check_path_components(attempt_root, relative, create_parents=False)
        _check_regular(path, missing_ok=False, kind="committed artifact")
        return path

    def read_artifact(self, ref: Mapping[str, Any], relative_path: str) -> bytes:
        """Read one validated artifact from an exact ``artifact_ref`` generation."""

        manifest = self._load_ref(ref)
        path = self.artifact_path(ref, relative_path)
        payload = _read_regular(path, kind="committed artifact")
        expected = next(entry for entry in manifest.artifacts if entry.path == _safe_relative_path(relative_path))
        if len(payload) != expected.size or _sha256_bytes(payload) != expected.sha256:
            _fail("artifact_hash_mismatch", f"committed artifact does not match manifest: {relative_path}")
        return payload


__all__ = [
    "ArtifactAttempt",
    "ArtifactEntry",
    "ArtifactStore",
    "ArtifactStoreFailure",
    "CommittedManifest",
    "DIRECTORY_MODE",
    "FILE_MODE",
    "SCHEMA_VERSION",
]
