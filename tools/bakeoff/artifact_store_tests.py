#!/usr/bin/env python3
"""Fault-injection and crash/retry tests for :mod:`artifact_store`."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
import threading
import unittest

try:
    from .artifact_store import ArtifactStore, ArtifactStoreFailure, FILE_MODE
except ImportError:  # pragma: no cover - direct air-gapped invocation
    from artifact_store import ArtifactStore, ArtifactStoreFailure, FILE_MODE  # type: ignore


IDENTITY = {
    "protocol_sha256": "1" * 64,
    "corpus_sha256": "2" * 64,
    "matrix_sha256": "3" * 64,
    "profile": "screen",
    "environment": "el8-x86_64",
}


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _raise_once(operation: str, *, path_name: str | None = None):
    fired = False

    def hook(actual: str, path: Path) -> None:
        nonlocal fired
        if not fired and actual == operation and (path_name is None or path.name == path_name):
            fired = True
            raise ArtifactStoreFailure(operation, f"injected {operation} failure at {path}")

    return hook


def _assert_failure(test: unittest.TestCase, callback, kind: str | None = None) -> None:
    with test.assertRaises(ArtifactStoreFailure) as context:
        callback()
    if kind is not None:
        test.assertEqual(context.exception.kind, kind)


class ArtifactStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="whitewater-artifacts-")
        # macOS exposes /var and /tmp as symlink aliases.  The store intentionally rejects
        # unresolved caller paths, so tests pass the canonical temporary root explicitly.
        self.root = Path(self.temporary.name).resolve()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _store(self, *, fault_hook=None) -> ArtifactStore:
        return ArtifactStore(self.root / "runs", IDENTITY, fault_hook=fault_hook)

    def _commit(self, store: ArtifactStore, cell: str, name: str, payload: bytes) -> dict:
        attempt = store.begin(cell)
        attempt.stage_bytes(name, payload)
        return attempt.commit()

    def _manifest(self, store: ArtifactStore, ref: dict) -> dict:
        return store.load_ref(ref)

    def test_identity_owned_layout_modes_and_unique_attempts(self) -> None:
        store = self._store()
        self.assertEqual(store.run_root.parent, store.parent)
        self.assertEqual(len(store.run_root.name), 64)
        self.assertEqual(stat.S_IMODE(store.run_root.joinpath("identity.json").stat().st_mode), FILE_MODE)
        self.assertEqual(stat.S_IMODE(store.cells_root.joinpath("..", "identity.json").stat().st_mode), FILE_MODE)

        first = store.begin("candidate/shot/a")
        second = store.begin("candidate/shot/a")
        self.assertNotEqual(first.attempt_id, second.attempt_id)
        self.assertTrue(first.root.is_dir())
        self.assertTrue(second.root.is_dir())
        self.assertTrue(first.root.parent == second.root.parent)
        self.assertIsNone(store.load("candidate/shot/a"))
        self.assertEqual(store.load_all(), {})

    def test_one_coordinating_writer_is_enforced_per_run(self) -> None:
        first = self._store()
        second = self._store()
        first.begin("cell-1")
        with self.assertRaises(ArtifactStoreFailure) as context:
            second.begin("cell-2")
        self.assertEqual(context.exception.kind, "writer_busy")
        first.close()
        second.begin("cell-2")

    def test_close_invalidates_old_attempts_before_a_new_store_acquires_the_lock(self) -> None:
        first = self._store()
        stale = first.begin("cell-1")
        first.close()

        second = self._store()
        current = second.begin("cell-1")
        current.stage_bytes("result.bin", b"current")

        # The old attempt still has a perfectly valid-looking directory, but its owning store
        # has released the run lock.  Every public attempt operation must observe that lifetime
        # boundary instead of writing into a run now owned by ``second``.
        for operation in (
            lambda: stale.stage_bytes("stale.bin", b"stale"),
            stale.validate,
            stale.commit,
            stale.reconcile,
        ):
            _assert_failure(self, operation, "store_closed")
        self.assertFalse((stale.root / "stale.bin").exists())

        current_ref = current.commit()
        self.assertEqual(second.read_artifact(current_ref, "result.bin"), b"current")
        second.close()

    def test_begin_setup_failure_releases_lock_and_requires_fresh_store(self) -> None:
        store = self._store()
        failed_path: Path | None = None

        def fail_setup_fsync(operation: str, path: Path) -> None:
            nonlocal failed_path
            if operation == "fsync" and path == store.cells_root:
                failed_path = path
                raise ArtifactStoreFailure("fsync", "injected begin directory fsync failure")

        store.fault_hook = fail_setup_fsync
        _assert_failure(self, lambda: store.begin("cell-1"), "fsync")
        self.assertEqual(failed_path, store.cells_root)
        self.assertTrue(store._closed)
        _assert_failure(self, lambda: store.begin("cell-2"), "store_closed")

        # The failed begin created part of the attempt tree before fsync failed.  A new store can
        # still acquire the same run's writer lock and publish a fresh UUID attempt safely.
        replacement_store = self._store()
        replacement = replacement_store.begin("cell-1")
        replacement.stage_bytes("result.bin", b"replacement")
        ref = replacement.commit()
        self.assertEqual(replacement_store.read_artifact(ref, "result.bin"), b"replacement")
        replacement_store.close()

    def test_stage_bytes_and_private_source_file_publish_valid_manifest(self) -> None:
        store = self._store()
        source = self.root / "source.bin"
        source.write_bytes(b"source bytes\n")
        os.chmod(source, 0o600)
        attempt = store.begin("cell-1")
        attempt.stage_bytes("result.json", b'{"ok":true}\n')
        attempt.stage_file("previews/preview.pfm", source)
        ref = attempt.commit()
        manifest = self._manifest(store, ref)

        self.assertEqual(manifest["cell_id"], "cell-1")
        self.assertEqual([entry["path"] for entry in manifest["artifacts"]], ["previews/preview.pfm", "result.json"])
        self.assertEqual(manifest["artifacts"][0]["sha256"], _sha256(b"source bytes\n"))
        for entry in manifest["artifacts"]:
            path = store.artifact_path(ref, entry["path"])
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), FILE_MODE)
        self.assertEqual(store.read_artifact(ref, "result.json"), b'{"ok":true}\n')
        self.assertEqual(store.load("cell-1"), manifest)
        self.assertEqual(store.load_all(), {"cell-1": manifest})
        with self.assertRaises(ArtifactStoreFailure):
            attempt.stage_bytes("late.bin", b"late")

    def test_abandoned_attempts_are_ignored_and_prior_commit_survives(self) -> None:
        store = self._store()
        old = self._commit(store, "cell-1", "result.json", b"old")
        old_manifest = self._manifest(store, old)
        committed_path = store._manifest_path("cell-1")
        old_manifest_bytes = committed_path.read_bytes()

        abandoned = store.begin("cell-1")
        abandoned.stage_bytes("result.json", b"partial retry")
        abandoned.stage_bytes("debug/partial.log", b"not committed")
        self.assertEqual(store.load("cell-1"), old_manifest)
        self.assertEqual(committed_path.read_bytes(), old_manifest_bytes)
        self.assertNotIn("debug/partial.log", json.dumps(store.load_all()))

        replacement = store.begin("cell-1")
        replacement.stage_bytes("result.json", b"new")
        new = replacement.commit()
        self.assertNotEqual(new["attempt_id"], old["attempt_id"])
        self.assertEqual(store.read_artifact(new, "result.json"), b"new")
        self.assertEqual(self._manifest(store, old), old_manifest)
        # The old attempt is immutable evidence and was not deleted as a side effect of retry.
        old_path = store.run_root / "cells" / old["cell_sha256"] / "attempts" / old["attempt_id"] / "result.json"
        self.assertEqual(old_path.read_bytes(), b"old")

    def test_exact_ref_never_follows_newer_current_pointer(self) -> None:
        store = self._store()
        old_ref = self._commit(store, "cell-1", "result.json", b"old")
        old_manifest = store.load_ref(old_ref)
        # This is the shape resume state will persist: an exact generation, not only a cell key.
        resume_record = {"cell_id": "cell-1", "artifact_ref": old_ref}

        new_ref = self._commit(store, "cell-1", "result.json", b"new")
        new_manifest = store.load_ref(new_ref)
        self.assertNotEqual(old_ref["manifest_sha256"], new_ref["manifest_sha256"])
        self.assertEqual(store.load("cell-1"), new_manifest)
        self.assertEqual(store.load_ref(resume_record["artifact_ref"]), old_manifest)
        self.assertEqual(store.read_artifact(resume_record["artifact_ref"], "result.json"), b"old")
        self.assertEqual(store.read_artifact(new_ref, "result.json"), b"new")

    def test_manifest_faults_never_clobber_previous_commit(self) -> None:
        for operation in ("write", "fsync", "replace", "manifest_publication"):
            with self.subTest(operation=operation):
                store = self._store()
                old = self._commit(store, "cell-1", "result.json", b"old")
                old_manifest = self._manifest(store, old)
                committed_path = store._manifest_path("cell-1")
                before = committed_path.read_bytes()
                retry = store.begin("cell-1")
                retry.stage_bytes("result.json", b"new")
                store.fault_hook = _raise_once(operation)
                _assert_failure(self, retry.commit, operation)
                store.fault_hook = None
                self.assertEqual(committed_path.read_bytes(), before)
                self.assertEqual(store.load("cell-1"), old_manifest)
                self.assertEqual(store.read_artifact(old, "result.json"), b"old")
                store.close()

    def test_stage_faults_leave_no_partial_final_file_and_do_not_touch_commit(self) -> None:
        for operation in ("write", "fsync", "replace"):
            with self.subTest(operation=operation):
                store = self._store()
                old = self._commit(store, "cell-1", "result.json", b"old")
                old_manifest = self._manifest(store, old)
                retry = store.begin("cell-1")
                store.fault_hook = _raise_once(operation)
                _assert_failure(self, lambda: retry.stage_bytes("new.bin", b"new"), operation)
                store.fault_hook = None
                self.assertFalse((retry.root / "new.bin").exists())
                self.assertEqual(store.load("cell-1"), old_manifest)
                store.close()

    def test_directory_fsync_after_replace_may_report_failure_but_leaves_valid_new_commit(self) -> None:
        store = self._store()
        old = self._commit(store, "cell-1", "result.json", b"old")
        old_manifest = self._manifest(store, old)
        retry = store.begin("cell-1")
        retry.stage_bytes("result.json", b"new")
        cell_root = store._manifest_path("cell-1").parent

        def fail_publication_directory(operation: str, path: Path) -> None:
            if operation == "fsync" and path == cell_root:
                raise ArtifactStoreFailure("fsync", "injected directory fsync failure")

        store.fault_hook = fail_publication_directory
        _assert_failure(self, retry.commit, "fsync")
        store.fault_hook = None
        loaded = store.load("cell-1")
        self.assertIsNotNone(loaded)
        self.assertNotEqual(loaded["attempt_id"], old["attempt_id"])
        new_ref = retry.reconcile()
        self.assertIsNotNone(new_ref)
        self.assertEqual(store.read_artifact(new_ref, "result.json"), b"new")
        self.assertEqual(store.load_ref(new_ref), loaded)
        self.assertEqual(store.load_ref(old), old_manifest)
        with self.assertRaises(ArtifactStoreFailure):
            retry.stage_bytes("late.bin", b"late")
        with self.assertRaises(ArtifactStoreFailure):
            retry.commit()
        old_path = store.run_root / "cells" / old["cell_sha256"] / "attempts" / old["attempt_id"] / "result.json"
        self.assertEqual(old_path.read_bytes(), b"old")

    def test_stage_post_replace_fsync_failure_poison_attempt(self) -> None:
        store = self._store()
        old = self._commit(store, "cell-1", "result.bin", b"old")
        retry = store.begin("cell-1")

        def fail_attempt_directory(operation: str, path: Path) -> None:
            if operation == "fsync" and path == retry.root:
                raise ArtifactStoreFailure("fsync", "injected stage directory fsync failure")

        store.fault_hook = fail_attempt_directory
        _assert_failure(self, lambda: retry.stage_bytes("result.bin", b"new"), "fsync")
        store.fault_hook = None
        self.assertEqual(store.load("cell-1"), self._manifest(store, old))
        # The replace may already have installed the staged file in this abandoned attempt, but
        # that attempt can no longer be mistaken for writable/resumable state.
        with self.assertRaises(ArtifactStoreFailure):
            retry.stage_bytes("late.bin", b"late")
        with self.assertRaises(ArtifactStoreFailure):
            retry.commit()

    def test_nested_artifact_parent_fsyncs_bottom_up_and_intermediate_failure_poison_attempt(self) -> None:
        store = self._store()
        old = self._commit(store, "cell-1", "result.bin", b"old")
        retry = store.begin("cell-1")
        intermediate = retry.root / "nested"
        seen: list[Path] = []

        def fail_intermediate(operation: str, path: Path) -> None:
            if operation == "fsync":
                seen.append(path)
                if path == intermediate:
                    raise ArtifactStoreFailure("fsync", "injected intermediate directory failure")

        store.fault_hook = fail_intermediate
        _assert_failure(self, lambda: retry.stage_bytes("nested/deeper/result.bin", b"new"), "fsync")
        store.fault_hook = None
        self.assertIn(retry.root / "nested" / "deeper", seen)
        self.assertIn(intermediate, seen)
        self.assertEqual(store.load("cell-1"), self._manifest(store, old))
        with self.assertRaises(ArtifactStoreFailure):
            retry.commit()

    def test_path_traversal_and_symlinks_are_rejected(self) -> None:
        store = self._store()
        attempt = store.begin("cell-1")
        for unsafe in ("../escape", "/absolute", "a/../b", "a//b", "a\\b", ""):
            with self.subTest(unsafe=unsafe):
                _assert_failure(self, lambda unsafe=unsafe: attempt.stage_bytes(unsafe, b"bad"), "path_safety")

        outside = self.root / "outside"
        outside.mkdir()
        (attempt.root / "linked").symlink_to(outside, target_is_directory=True)
        _assert_failure(self, lambda: attempt.stage_bytes("linked/escape.bin", b"bad"), "symlink_path")

        manifest_path = store._manifest_path("cell-2")
        manifest_path.symlink_to(outside / "manifest")
        second = store.begin("cell-2")
        second.stage_bytes("result.bin", b"result")
        _assert_failure(self, second.commit, "symlink_path")
        self.assertFalse((outside / "manifest").exists())

    def test_loader_revalidates_committed_artifacts_and_rejects_tampering(self) -> None:
        store = self._store()
        ref = self._commit(store, "cell-1", "result.bin", b"original")
        fabricated_manifest = store.load_ref(ref)
        artifact = store.artifact_path(ref, "result.bin")
        artifact.write_bytes(b"tampered")
        os.chmod(artifact, FILE_MODE)
        _assert_failure(self, lambda: store.load("cell-1"), "artifact_hash_mismatch")

        with self.assertRaises(ArtifactStoreFailure) as context:
            store.artifact_path(fabricated_manifest, "result.bin")
        self.assertEqual(context.exception.kind, "artifact_ref_shape")

    def test_identity_and_current_ref_are_defensive_copies(self) -> None:
        identity = {"nested": {"values": [1, 2]}, "label": "stable"}
        store = ArtifactStore(self.root / "runs", identity)
        identity["nested"]["values"].append(3)
        self.assertEqual(store.identity["nested"]["values"], [1, 2])

        exposed = store.identity
        exposed["nested"]["values"].append(4)
        self.assertEqual(store.identity["nested"]["values"], [1, 2])

        ref = self._commit(store, "cell-1", "result.bin", b"result")
        current = store.current_ref("cell-1")
        self.assertEqual(current, ref)
        current["attempt_id"] = "changed"
        self.assertEqual(store.current_ref("cell-1"), ref)

    def test_invalid_utf8_and_surrogates_fail_at_input_boundaries(self) -> None:
        with self.assertRaises(ArtifactStoreFailure) as context:
            ArtifactStore(self.root / "invalid-identity", {"bad": "\ud800"})
        self.assertEqual(context.exception.kind, "text_encoding")

        store = self._store()
        with self.assertRaises(ArtifactStoreFailure) as context:
            store.begin("cell-\ud800")
        self.assertEqual(context.exception.kind, "text_encoding")
        attempt = store.begin("cell-valid")
        with self.assertRaises(ArtifactStoreFailure) as context:
            attempt.stage_bytes("preview-\ud800.pfm", b"bad")
        self.assertEqual(context.exception.kind, "text_encoding")

    def test_loader_reads_only_committed_manifests_after_reopen(self) -> None:
        store = self._store()
        abandoned = store.begin("abandoned")
        abandoned.stage_bytes("result.bin", b"partial")
        committed = self._commit(store, "committed", "result.bin", b"complete")
        committed_manifest = self._manifest(store, committed)

        reopened = ArtifactStore(self.root / "runs", IDENTITY)
        self.assertIsNone(reopened.load("abandoned"))
        self.assertEqual(reopened.load("committed"), committed_manifest)
        self.assertEqual(reopened.load_ref(committed), committed_manifest)
        self.assertEqual(reopened.load_all(), {"committed": committed_manifest})

    def test_identity_collision_and_symlinked_parent_fail_closed(self) -> None:
        store = self._store()
        identity_path = store.run_root / "identity.json"
        identity_path.write_text(json.dumps({"bad": True}), encoding="utf-8")
        os.chmod(identity_path, FILE_MODE)
        with self.assertRaises(ArtifactStoreFailure) as context:
            ArtifactStore(self.root / "runs", IDENTITY)
        self.assertIn(context.exception.kind, {"identity_shape", "identity_mismatch"})

        symlink_parent = self.root / "symlink-parent"
        symlink_parent.symlink_to(self.root / "runs", target_is_directory=True)
        with self.assertRaises(ArtifactStoreFailure) as context:
            ArtifactStore(symlink_parent, IDENTITY)
        self.assertEqual(context.exception.kind, "symlink_path")

    def test_initial_directory_entries_are_fsynced_bottom_up(self) -> None:
        seen: list[Path] = []

        def record_fsync(operation: str, path: Path) -> None:
            if operation == "fsync":
                seen.append(path)

        store = ArtifactStore(self.root / "durability-runs", IDENTITY, fault_hook=record_fsync)
        try:
            self.assertIn(store.parent, seen)
            self.assertIn(store.run_root, seen)
            self.assertLess(seen.index(store.parent), seen.index(store.run_root))
        finally:
            store.close()

    def test_private_identity_temps_are_reconciled_but_unknown_content_is_rejected(self) -> None:
        identity_sha256 = _sha256(
            json.dumps(IDENTITY, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        stale_parent = self.root / "stale-identity-runs"
        stale_root = stale_parent / identity_sha256
        stale_root.mkdir(parents=True)
        (stale_root / ".identity.json.crashed.tmp").write_bytes(b"partial identity")

        store = ArtifactStore(stale_parent, IDENTITY)
        self.assertTrue((stale_root / "identity.json").is_file())
        store.close()

        unknown_parent = self.root / "unknown-identity-runs"
        unknown_root = unknown_parent / identity_sha256
        unknown_root.mkdir(parents=True)
        (unknown_root / "unrelated.tmp").write_bytes(b"not a private identity temp")
        with self.assertRaises(ArtifactStoreFailure) as context:
            ArtifactStore(unknown_parent, IDENTITY)
        self.assertEqual(context.exception.kind, "identity_missing")

    def test_same_identity_constructor_race_does_not_treat_private_temp_as_unknown(self) -> None:
        parent = self.root / "constructor-race-runs"
        temp_started = threading.Event()
        release_temp = threading.Event()
        first_store: list[ArtifactStore] = []
        first_errors: list[BaseException] = []

        def pause_identity_write(operation: str, path: Path) -> None:
            if operation == "write" and path.name.startswith(".identity.json."):
                temp_started.set()
                release_temp.wait(timeout=5)

        def construct_first() -> None:
            try:
                first_store.append(ArtifactStore(parent, IDENTITY, fault_hook=pause_identity_write))
            except BaseException as exc:  # noqa: BLE001 - preserve thread failure for assertion
                first_errors.append(exc)

        thread = threading.Thread(target=construct_first)
        thread.start()
        second_store: ArtifactStore | None = None
        try:
            self.assertTrue(temp_started.wait(timeout=5))
            second_store = ArtifactStore(parent, IDENTITY)
        finally:
            release_temp.set()
            thread.join(timeout=5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(first_errors, [])
        self.assertEqual(len(first_store), 1)
        if second_store is not None:
            second_store.close()
        first_store[0].close()

    def test_symlinked_ancestor_is_rejected_before_any_target_write(self) -> None:
        real_parent = self.root / "real-parent"
        real_parent.mkdir()
        linked_parent = self.root / "linked-parent"
        linked_parent.symlink_to(real_parent, target_is_directory=True)
        redirected_root = real_parent / "nested"

        with self.assertRaises(ArtifactStoreFailure) as context:
            ArtifactStore(linked_parent / "nested", IDENTITY)
        self.assertEqual(context.exception.kind, "symlink_path")
        self.assertFalse(redirected_root.exists())

    def test_dotdot_cannot_hide_a_symlinked_ancestor(self) -> None:
        real_parent = self.root / "dotdot-real-parent"
        real_parent.mkdir()
        linked_parent = self.root / "dotdot-linked-parent"
        linked_parent.symlink_to(real_parent, target_is_directory=True)

        normalized_target = self.root / "dotdot-nested"
        symlink_target = real_parent / "dotdot-nested"
        with self.assertRaises(ArtifactStoreFailure) as context:
            ArtifactStore(linked_parent / ".." / "dotdot-nested", IDENTITY)
        self.assertEqual(context.exception.kind, "symlink_path")
        self.assertFalse(normalized_target.exists())
        self.assertFalse(symlink_target.exists())


if __name__ == "__main__":
    unittest.main()
