#!/usr/bin/env python3
"""Acquire the exact official original-RAFT checkpoint member.

The upstream repository points at an author-linked archive rather than a release asset.  This
small, dependency-free tool makes that indirection reproducible: it verifies the archive bytes,
extracts exactly one regular ZIP member, verifies the member size and SHA256, and publishes a
mode-0644 checkpoint atomically.  Network access is opt-in; an exporter never downloads a
checkpoint implicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import stat
import tempfile
import urllib.request
import zipfile

try:
    from artifact_workflow import (  # type: ignore  # pylint: disable=wrong-import-position
        ArtifactError,
        EXPECTED_MODE,
        require_regular_mode,
        sha256_file,
    )
except ModuleNotFoundError:  # pragma: no cover - package import path
    from .artifact_workflow import (  # type: ignore
        ArtifactError,
        EXPECTED_MODE,
        require_regular_mode,
        sha256_file,
    )


ARCHIVE_URL = "https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip"
ARCHIVE_SHA256 = "4be6101b271f58ec49866da5cf609fd17e86e9cae2483f70630ef4a295dc66bd"
MEMBER_NAME = "models/raft-things.pth"
MEMBER_SIZE = 21108000
MEMBER_SHA256 = "fcfa4125d6418f4de95d84aec20a3c5f4e205101715a79f193243c186ac9a7e1"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def verify_archive(path: Path, *, expected_sha256: str = ARCHIVE_SHA256) -> None:
    """Verify the exact official archive before opening its member."""

    require_regular_mode(path, "RAFT model archive")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ArtifactError(
            f"RAFT archive SHA256 is {actual}, expected {expected_sha256}: {path}"
        )


def _member_is_regular(info: zipfile.ZipInfo) -> bool:
    mode = (info.external_attr >> 16) & 0o177777
    file_type = stat.S_IFMT(mode)
    # ZIPs produced by common tools leave the mode unset.  If it is present, reject every
    # non-regular member so an archive cannot smuggle a symlink into the checkpoint path.
    return file_type in (0, stat.S_IFREG)


def extract_verified_member(
    archive_path: Path,
    output_path: Path,
    *,
    member_name: str = MEMBER_NAME,
    expected_size: int = MEMBER_SIZE,
    expected_sha256: str = MEMBER_SHA256,
    expected_archive_sha256: str = ARCHIVE_SHA256,
) -> None:
    """Verify and atomically extract one checkpoint member.

    The expected values are parameters to keep this helper unit-testable; production callers
    use the module constants above.  Duplicate member names are rejected rather than silently
    selecting whichever ZIP entry a reader happens to return.
    """

    verify_archive(archive_path, expected_sha256=expected_archive_sha256)
    try:
        with zipfile.ZipFile(archive_path, "r") as archive:
            matches = [info for info in archive.infolist() if info.filename == member_name]
            if len(matches) != 1:
                raise ArtifactError(
                    f"RAFT archive must contain exactly one {member_name!r}; found {len(matches)}"
                )
            info = matches[0]
            if not _member_is_regular(info):
                raise ArtifactError(f"RAFT checkpoint member is not a regular file: {member_name}")
            if info.file_size != expected_size:
                raise ArtifactError(
                    f"RAFT checkpoint member is {info.file_size} bytes, expected {expected_size}"
                )
            payload = archive.read(info)
    except zipfile.BadZipFile as exc:
        raise ArtifactError(f"RAFT archive is not a valid ZIP: {archive_path}") from exc

    actual = _sha256_bytes(payload)
    if actual != expected_sha256:
        raise ArtifactError(
            f"RAFT checkpoint member SHA256 is {actual}, expected {expected_sha256}"
        )

    # ``absolute()`` normalizes the parent without following a final symlink; the lstat below
    # must see a caller-supplied symlink so it can be rejected by the public-file gate.
    output_path = output_path.absolute()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        output_path.lstat()
    except FileNotFoundError:
        pass
    else:
        # Never overwrite a different or weakly-permissioned file.  A verified existing output
        # is idempotent and is accepted only if it has the exact public-file mode and bytes.
        require_regular_mode(output_path, "RAFT checkpoint output")
        if output_path.stat().st_size != expected_size or sha256_file(output_path) != expected_sha256:
            raise ArtifactError(f"RAFT checkpoint output already exists with different bytes: {output_path}")
        return

    with tempfile.NamedTemporaryFile(
        prefix=output_path.name + ".", suffix=".tmp", dir=output_path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        stream.write(payload)
    try:
        temporary.chmod(EXPECTED_MODE)
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)


def _path_exists_without_following_symlink(path: Path) -> bool:
    """Return whether a path exists, including a dangling symlink."""

    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def download_archive(output_path: Path, *, expected_sha256: str = ARCHIVE_SHA256) -> None:
    """Download the pinned archive without silently replacing an existing copy.

    An existing mode-0644 file is accepted only when it is the exact verified archive.  This
    makes an explicit ``--archive-copy`` idempotent while preserving a different file rather
    than clobbering it.  The link-at-publish step also keeps that no-clobber property if another
    process creates the destination after the initial check.
    """

    output_path = output_path.absolute()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if _path_exists_without_following_symlink(output_path):
        verify_archive(output_path, expected_sha256=expected_sha256)
        return

    with tempfile.NamedTemporaryFile(
        prefix=output_path.name + ".", suffix=".tmp", dir=output_path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        with urllib.request.urlopen(ARCHIVE_URL) as response:  # nosec B310 - fixed HTTPS URL
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                stream.write(chunk)
    try:
        temporary.chmod(EXPECTED_MODE)
        verify_archive(temporary, expected_sha256=expected_sha256)
        try:
            # ``os.link`` publishes without replacing a destination that appeared during the
            # download.  The temporary and destination are in the same directory, so this is
            # atomic on the filesystems supported by the exporter.
            os.link(temporary, output_path)
        except FileExistsError:
            # A concurrent publisher may have won the race.  Treat an exact copy as the same
            # idempotent result and reject any other bytes or mode.
            verify_archive(output_path, expected_sha256=expected_sha256)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--archive", type=Path, help="already-downloaded official models.zip")
    source.add_argument(
        "--download",
        action="store_true",
        help=f"download the pinned official archive URL ({ARCHIVE_URL})",
    )
    parser.add_argument("--output", required=True, type=Path, help="destination for raft-things.pth")
    parser.add_argument(
        "--archive-copy",
        type=Path,
        help="when downloading, also retain the verified archive at this path",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.archive is not None and args.archive_copy is not None:
        raise ArtifactError("--archive-copy is only valid with --download")

    archive = args.archive
    if args.download:
        if args.archive_copy is not None:
            archive = args.archive_copy.absolute()
            download_archive(archive)
            extract_verified_member(archive, args.output)
            print(f"archive:     {archive}")
        else:
            # A downloaded archive is an implementation detail unless the caller explicitly
            # asks for --archive-copy.  Keep it outside cwd and remove it after extraction so a
            # routine acquisition cannot leave a large, mutable models.zip behind.
            with tempfile.TemporaryDirectory(prefix="whitewater-raft-archive-") as directory:
                archive = Path(directory) / "models.zip"
                download_archive(archive)
                extract_verified_member(archive, args.output)
                print(f"archive:     {archive} (temporary; removed after extraction)")
            archive = None

    if args.download:
        # The download branches above perform extraction and print the archive provenance while
        # the temporary path (if any) still exists.  The common member output is printed here.
        print(f"archive_sha: {ARCHIVE_SHA256}")
        print(f"member:      {args.output.absolute()}")
        print(f"member_sha:  {MEMBER_SHA256}")
        return 0

    assert archive is not None
    extract_verified_member(archive.absolute(), args.output)
    print(f"archive:     {archive.absolute()}")
    print(f"archive_sha: {ARCHIVE_SHA256}")
    print(f"member:      {args.output.absolute()}")
    print(f"member_sha:  {MEMBER_SHA256}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ArtifactError, OSError, ValueError, zipfile.BadZipFile) as exc:
        print(f"fetch_raft_checkpoint.py: error: {exc}")
        raise SystemExit(1)
