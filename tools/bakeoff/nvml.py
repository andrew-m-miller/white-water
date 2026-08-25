#!/usr/bin/env python3
"""P25-6 CUDA device-memory sampling for the airgapped target measurement.

The bake-off runner must record device memory across a cell's lifecycle: baseline,
session-create peak, warm/steady peak, cleanup and process-exit, both idle and beside a live
Flame Batch workload (see ``docs/phase2.5-implementation-plan.md``, "P25-6 - Airgapped target
measurement"). This module is the small, dependency-injected boundary between that measurement
and two consumers:

- the frozen ``resource``/``nvml_sample`` shapes bound by ``bakeoff/report-v2.schema.json``
  (``$defs/resource`` and ``$defs/nvml_sample``), embedded per result in ``report.json``; and
- the operator's separate ``nvml.csv``, a continuous/aggregated device log across the whole
  run, keyed by cell identity, stage and sample index.

The sampler is driven entirely through an injected :class:`NvmlBackend`, so importing this
module and running its unit tests requires neither ``pynvml`` nor a GPU. Only
:class:`PynvmlBackend` touches the real driver, and it imports ``pynvml`` lazily so the module
stays importable on a machine that lacks it (this development machine is macOS with neither).

Boundary snapshots (:meth:`NvmlSampler.sample`) are point-in-time and can miss a transient
allocation made and freed between two calls; :meth:`NvmlSampler.poll` continuously polls a
work window on a background thread and records the observed maximum instead, which is what
P25-6's session-create/warm/steady *peak* requirement needs (see ``docs/context.md``, Session
6: "boundary-sampled NVML does not capture the rejected allocation").
"""

from __future__ import annotations

import csv
import math
import os
import stat
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence


class NvmlFailure(ValueError):
    """Stable, reportable NVML sampling/reporting failure."""

    def __init__(self, kind: str, message: str):
        self.kind = kind
        self.reason = kind
        self.failure_type = "nvml_failure"
        self.message = message
        super().__init__(f"{kind}: {message}")


def _fail(kind: str, message: str) -> None:
    raise NvmlFailure(kind, message)


# Enum bound by bakeoff/report-v2.schema.json $defs/nvml_sample.properties.stage.
STAGES: tuple[str, ...] = ("baseline", "session_create", "steady", "cleanup", "process_exit", "other")

_IDENTITY_FIELDS = ("candidate_id", "shot_id", "conditioning_token", "cap_token", "provider", "host_load")

# Frozen nvml.csv header. This is the operator's continuous device-wide log across an entire
# run, distinct from the per-cell "nvml_samples" array embedded in report.json. Columns are
# appended, never reordered or removed, matching the "choice option order is API" discipline
# used elsewhere in this repo for saved-setup-shaped surfaces.
NVML_CSV_HEADER: tuple[str, ...] = (
    "candidate_id", "shot_id", "conditioning_token", "cap_token", "provider", "host_load",
    "stage", "sample_index", "timestamp_unix_s", "device_used_mib", "process_used_mib",
)


class NvmlBackend(Protocol):
    """Injectable GPU-memory query backend used by :class:`NvmlSampler`.

    All memory quantities returned by a backend are MiB (mebibytes), matching the report
    schema's ``*_mib`` fields. A backend resolves one device handle per index and is expected
    to reuse it for repeated queries; :class:`NvmlSampler` calls ``device_handle`` exactly once.
    """

    def device_handle(self, device_index: int) -> Any:
        """Resolve a stable handle for one CUDA device index."""
        ...

    def device_used_mib(self, handle: Any) -> float:
        """Return device-wide used memory in MiB for the resolved handle."""
        ...

    def process_used_mib(self, handle: Any, pid: int) -> float | None:
        """Return this process's used memory in MiB, or None when unavailable.

        None means the driver/NVML build does not expose per-process accounting for this
        device (e.g. no compute-process enumeration support), not that usage is zero.
        """
        ...


class PynvmlBackend:
    """Real :class:`NvmlBackend` backed by ``pynvml``.

    ``pynvml`` is imported lazily in ``__init__`` so that importing this module, and running
    its unit tests with an injected fake backend, never requires the dependency or a GPU. Use
    as a context manager, or call :meth:`close` explicitly, to shut NVML down cleanly.
    """

    def __init__(self) -> None:
        try:
            import pynvml  # type: ignore
        except ImportError as exc:
            raise NvmlFailure("runtime_error", f"pynvml is not installed: {exc}") from exc
        self._pynvml = pynvml
        self._initialized = False
        try:
            pynvml.nvmlInit()
        except Exception as exc:  # pynvml raises its own NVMLError hierarchy
            raise NvmlFailure("runtime_error", f"NVML initialization failed: {exc}") from exc
        self._initialized = True

    def __enter__(self) -> "PynvmlBackend":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        """Shut NVML down. Safe to call more than once."""

        if self._initialized:
            try:
                self._pynvml.nvmlShutdown()
            finally:
                self._initialized = False

    def device_handle(self, device_index: int) -> Any:
        try:
            return self._pynvml.nvmlDeviceGetHandleByIndex(device_index)
        except Exception as exc:
            raise NvmlFailure("device_unavailable", f"cannot resolve NVML device {device_index}: {exc}") from exc

    def device_used_mib(self, handle: Any) -> float:
        try:
            info = self._pynvml.nvmlDeviceGetMemoryInfo(handle)
        except Exception as exc:
            raise NvmlFailure("query_failed", f"cannot query NVML device memory: {exc}") from exc
        return info.used / (1024.0 * 1024.0)

    def process_used_mib(self, handle: Any, pid: int) -> float | None:
        processes: list[Any] = []
        # Compute and graphics process accounting are separate, driver-dependent NVML
        # capabilities; either or both may be unsupported for a given device/driver build.
        for getter in ("nvmlDeviceGetComputeRunningProcesses", "nvmlDeviceGetGraphicsRunningProcesses"):
            try:
                processes.extend(getattr(self._pynvml, getter)(handle))
            except Exception:
                continue
        for process in processes:
            if getattr(process, "pid", None) != pid:
                continue
            used = getattr(process, "usedGpuMemory", None)
            if used is None:
                return None
            return used / (1024.0 * 1024.0)
        return None


def _nonnegative_finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail("invalid_measurement", f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        _fail("invalid_measurement", f"{label} must be finite")
    if number < 0.0:
        _fail("invalid_measurement", f"{label} must be non-negative")
    return number


class NvmlSampler:
    """Accumulates ``nvml_sample`` records for one cell and reduces them to a ``resource``.

    Construction resolves the device handle once via the injected backend. There are two ways
    to record a sample:

    - :meth:`sample` takes one instantaneous boundary reading -- appropriate for ``baseline``,
      ``cleanup`` and ``process_exit``, which are naturally point-in-time.
    - :meth:`poll` continuously polls in the background across a ``session_create`` or
      ``steady`` work window and records the MAXIMUM observed reading as that stage's sample.
      Boundary-only snapshots miss a transient allocation that is made and freed between two
      ``sample()`` calls -- ``docs/context.md`` (Session 6) records exactly this gap: "boundary-
      sampled NVML does not capture the rejected allocation." P25-6 requires session-create and
      warm/steady *peaks*, so those two stages should be measured with :meth:`poll`, not
      :meth:`sample`.

    Both methods append to the same ordered sample list, so :meth:`resource` reduces over
    whichever mix of boundary and polled readings the caller took. If more than one ``baseline``
    sample is taken, the first is authoritative for :meth:`resource` -- callers should sample
    ``baseline`` exactly once per cell.
    """

    def __init__(
        self,
        backend: NvmlBackend,
        device_index: int,
        pid: int | None = None,
        *,
        clock: Callable[[], float] = time.time,
        poll_interval_s: float = 0.05,
    ) -> None:
        if not isinstance(poll_interval_s, (int, float)) or isinstance(poll_interval_s, bool) or poll_interval_s < 0:
            _fail("invalid_interval", "poll_interval_s must be a non-negative number")
        self._backend = backend
        self._device_index = device_index
        self._pid = os.getpid() if pid is None else pid
        self._clock = clock
        self._poll_interval_s = float(poll_interval_s)
        self._handle = backend.device_handle(device_index)
        self._samples: list[dict[str, Any]] = []
        self._timestamps: list[float] = []

    @property
    def device_index(self) -> int:
        return self._device_index

    @property
    def pid(self) -> int:
        return self._pid

    def _query_backend(self) -> tuple[float, float | None]:
        """One backend query, validated and unit-normalized. Shared by sample() and poll()."""

        used_mib = _nonnegative_finite(self._backend.device_used_mib(self._handle), "used_mib")
        process_used_mib = self._backend.process_used_mib(self._handle, self._pid)
        if process_used_mib is not None:
            process_used_mib = _nonnegative_finite(process_used_mib, "process_used_mib")
        return used_mib, process_used_mib

    def sample(self, stage: str) -> dict[str, Any]:
        """Query the backend once and record one schema-shaped ``nvml_sample``.

        ``process_used_mib`` is omitted entirely (never set to null) when the backend reports
        per-process accounting as unavailable, so the record always validates against the
        report schema's ``additionalProperties: false`` object. This is an instantaneous
        snapshot; use :meth:`poll` for a stage where a transient peak must not be missed.
        """

        if not isinstance(stage, str) or stage not in STAGES:
            _fail("unknown_stage", f"stage {stage!r} is not a permitted nvml_sample stage")
        timestamp = _nonnegative_finite(self._clock(), "timestamp_unix_s")
        used_mib, process_used_mib = self._query_backend()
        record: dict[str, Any] = {"stage": stage, "used_mib": used_mib}
        if process_used_mib is not None:
            record["process_used_mib"] = process_used_mib
        self._samples.append(record)
        self._timestamps.append(timestamp)
        return dict(record)

    def poll(self, stage: str, *, interval_s: float | None = None) -> "PollWindow":
        """Continuously poll memory during a work window and record the stage's MAXIMUM.

        Use as a context manager around the exact work being measured::

            with sampler.poll("session_create"):
                session = create_session(...)

        On ``__exit__`` this records exactly one schema-shaped ``nvml_sample`` for ``stage``
        whose ``used_mib``/``process_used_mib`` are the maximum seen across the window,
        including one reading taken synchronously and immediately on ``__enter__`` (before the
        background poller starts), so a window always yields at least one real reading even if
        it closes before the background thread runs. Polling continues on a background thread
        at ``interval_s`` (default: this sampler's configured ``poll_interval_s``) cadence, with
        the shared running maximum guarded by a lock. The poller always stops from ``__exit__``
        -- including when the polled block raises -- and whatever maximum it captured is still
        recorded; the caller's exception then propagates unchanged.
        """

        if not isinstance(stage, str) or stage not in STAGES:
            _fail("unknown_stage", f"stage {stage!r} is not a permitted nvml_sample stage")
        resolved_interval = self._poll_interval_s if interval_s is None else interval_s
        if not isinstance(resolved_interval, (int, float)) or isinstance(resolved_interval, bool) or resolved_interval < 0:
            _fail("invalid_interval", "poll interval_s must be a non-negative number")
        return PollWindow(self, stage, float(resolved_interval))

    def _record_polled_sample(self, stage: str, used_mib: float, process_used_mib: float | None) -> dict[str, Any]:
        timestamp = _nonnegative_finite(self._clock(), "timestamp_unix_s")
        record: dict[str, Any] = {"stage": stage, "used_mib": used_mib}
        if process_used_mib is not None:
            record["process_used_mib"] = process_used_mib
        self._samples.append(record)
        self._timestamps.append(timestamp)
        return dict(record)

    @property
    def samples(self) -> list[dict[str, Any]]:
        """All recorded ``nvml_sample`` records, in sample order."""

        return [dict(sample) for sample in self._samples]

    def resource(self) -> dict[str, Any]:
        """Reduce recorded samples to one ``resource`` object.

        Raises a typed failure if no ``baseline`` sample was recorded: the contract's
        incremental peak is meaningless without one.
        """

        if not self._samples:
            _fail("no_samples", "resource requires at least one recorded nvml sample")
        baseline = next((sample for sample in self._samples if sample["stage"] == "baseline"), None)
        if baseline is None:
            _fail("missing_baseline", "resource requires a baseline sample; incremental peak is undefined without one")
        baseline_used_mib = baseline["used_mib"]
        peak_used_mib = max(sample["used_mib"] for sample in self._samples)
        peak_incremental_mib = max(0.0, peak_used_mib - baseline_used_mib)
        resource: dict[str, Any] = {
            "peak_incremental_device_memory_gib": peak_incremental_mib / 1024.0,
            "baseline_device_memory_mib": baseline_used_mib,
            "peak_device_memory_mib": peak_used_mib,
            "nvml_samples": self.samples,
        }
        cleanup = next((sample for sample in reversed(self._samples) if sample["stage"] == "cleanup"), None)
        if cleanup is not None:
            resource["cleanup_device_memory_mib"] = cleanup["used_mib"]
        process_exit = next((sample for sample in reversed(self._samples) if sample["stage"] == "process_exit"), None)
        if process_exit is not None:
            resource["process_exit_device_memory_mib"] = process_exit["used_mib"]
        return resource

    def csv_rows(self, identity: Mapping[str, str]) -> list[list[str]]:
        """Render every recorded sample as ``nvml.csv`` data rows (no header) for one cell."""

        normalized_identity = _validate_csv_identity(identity)
        rows: list[list[str]] = []
        for index, (sample, timestamp) in enumerate(zip(self._samples, self._timestamps)):
            rows.append(_nvml_csv_row(normalized_identity, sample, index, timestamp))
        return rows


class PollWindow:
    """Context manager returned by :meth:`NvmlSampler.poll`.

    A background thread repeatedly queries the sampler's backend and tracks the running
    maximum ``used_mib``/``process_used_mib`` under a lock, so it is safe for the caller's
    thread to read no shared state at all -- the maximum is only read back, once, in
    ``__exit__``, after the poller has been stopped and joined. Not intended to be constructed
    directly; use :meth:`NvmlSampler.poll`.
    """

    def __init__(self, sampler: "NvmlSampler", stage: str, interval_s: float) -> None:
        self._sampler = sampler
        self._stage = stage
        self._interval_s = interval_s
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._reading_event = threading.Event()
        self._max_used_mib: float | None = None
        self._max_process_used_mib: float | None = None
        self._reading_count = 0
        self._background_error: BaseException | None = None
        self._thread: threading.Thread | None = None

    @property
    def reading_count(self) -> int:
        """Number of backend readings taken so far (thread-safe; mainly for tests)."""

        with self._lock:
            return self._reading_count

    def wait_for_reading_count(self, count: int, timeout: float = 2.0) -> None:
        """Block until at least ``count`` readings have been recorded.

        A synchronization helper (not required for normal use) that lets a test observe a
        specific number of background readings deterministically, without depending on the
        real poll interval: it blocks on an event the poller signals after each reading is
        fully recorded (max already updated under the lock), rather than guessing a sleep
        duration. Raises a typed failure if ``count`` is not reached within ``timeout``.
        """

        deadline = time.monotonic() + timeout
        while True:
            with self._lock:
                if self._reading_count >= count:
                    return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _fail("timeout", f"poll window did not reach {count} readings within {timeout}s")
            self._reading_event.wait(remaining)
            self._reading_event.clear()

    def _record_reading(self) -> None:
        used_mib, process_used_mib = self._sampler._query_backend()
        with self._lock:
            self._reading_count += 1
            if self._max_used_mib is None or used_mib > self._max_used_mib:
                self._max_used_mib = used_mib
            if process_used_mib is not None and (
                self._max_process_used_mib is None or process_used_mib > self._max_process_used_mib
            ):
                self._max_process_used_mib = process_used_mib
        self._reading_event.set()

    def _run(self) -> None:
        try:
            # wait() first: the immediate reading was already taken synchronously in
            # __enter__, so the first background reading is one interval later.
            while not self._stop_event.wait(self._interval_s):
                self._record_reading()
        except BaseException as exc:  # pragma: no cover - real backend failures only
            with self._lock:
                self._background_error = exc

    def __enter__(self) -> "PollWindow":
        self._record_reading()
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"nvml-poll-{self._stage}")
        self._thread.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        with self._lock:
            max_used_mib = self._max_used_mib
            max_process_used_mib = self._max_process_used_mib
            background_error = self._background_error
        # __enter__ always takes one synchronous reading before returning, so a window that
        # reaches __exit__ at all -- with or without a caller exception -- always has a max.
        assert max_used_mib is not None
        self._sampler._record_polled_sample(self._stage, max_used_mib, max_process_used_mib)
        if exc_type is None and background_error is not None:
            raise NvmlFailure("query_failed", f"nvml poll for stage {self._stage!r} failed: {background_error}") from background_error


def _validate_csv_identity(identity: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(identity, Mapping):
        _fail("identity_shape", "nvml.csv identity must be an object")
    if set(identity) != set(_IDENTITY_FIELDS):
        _fail("identity_shape", "nvml.csv identity must contain exactly the six cell identity fields")
    for field in _IDENTITY_FIELDS:
        if not isinstance(identity[field], str) or not identity[field]:
            _fail("identity_shape", f"nvml.csv identity.{field} must be a non-empty string")
    return {field: identity[field] for field in _IDENTITY_FIELDS}


def _csv_number(value: float) -> str:
    return repr(float(value))


def _nvml_csv_row(identity: Mapping[str, str], sample: Mapping[str, Any], sample_index: int, timestamp_unix_s: float) -> list[str]:
    return [
        *(identity[field] for field in _IDENTITY_FIELDS),
        sample["stage"],
        str(sample_index),
        _csv_number(timestamp_unix_s),
        _csv_number(sample["used_mib"]),
        _csv_number(sample["process_used_mib"]) if "process_used_mib" in sample else "",
    ]


def _check_destination_for_create(path: Path) -> None:
    try:
        path.lstat()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise NvmlFailure("output_path", str(exc)) from exc
    _fail("output_exists", f"nvml.csv already exists, use append_nvml_csv to extend it: {path}")


def _check_destination_for_append(path: Path) -> None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        _fail("missing_output", f"nvml.csv does not exist for append: {path}")
    except OSError as exc:
        raise NvmlFailure("output_path", str(exc)) from exc
    if stat.S_ISLNK(info.st_mode):
        _fail("symlink_output", f"nvml.csv symlink is not permitted: {path}")
    if not stat.S_ISREG(info.st_mode):
        _fail("nonregular_output", f"nvml.csv must be a regular file: {path}")
    if stat.S_IMODE(info.st_mode) != 0o644:
        _fail("output_mode", f"nvml.csv mode must be exactly 0644: {path}")


def _render_csv_lines(rows: Sequence[Sequence[str]], *, header: bool) -> bytes:
    from io import StringIO

    output = StringIO(newline="")
    writer = csv.writer(output, delimiter=",", quotechar='"', lineterminator="\n")
    if header:
        writer.writerow(NVML_CSV_HEADER)
    for row in rows:
        writer.writerow(row)
    return output.getvalue().encode("utf-8")


def write_nvml_csv(path: Path | str, rows: Sequence[Sequence[str]]) -> None:
    """Create ``nvml.csv`` with the frozen header, mode 0644, refusing to clobber an existing file.

    Staged in a same-directory temporary file and published with a no-clobber hard link, the
    same discipline :mod:`tools.bakeoff.reporting` uses for ``report.json``/``report.csv``.
    """

    destination = Path(path)
    _check_destination_for_create(destination)
    if not destination.parent.is_dir():
        _fail("output_path", f"nvml.csv parent is not a directory: {destination.parent}")
    payload = _render_csv_lines(rows, header=True)
    descriptor = -1
    temporary: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent))
        temporary = Path(name)
        os.fchmod(descriptor, 0o644)
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination, follow_symlinks=False)
        except FileExistsError:
            _fail("output_exists", f"nvml.csv appeared during publication: {destination}")
        os.unlink(temporary)
        temporary = None
        directory_descriptor = os.open(str(destination.parent), os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except OSError as exc:
        raise NvmlFailure("atomic_write", str(exc)) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def append_nvml_csv(path: Path | str, rows: Sequence[Sequence[str]]) -> None:
    """Append data rows to an existing ``nvml.csv``.

    The destination must already exist (created with :func:`write_nvml_csv`), be a regular
    file, mode 0644, and not a symlink. Rows are appended with a single ``O_APPEND`` write and
    an ``fsync``, which is atomic for one writer but is not a substitute for cross-process
    locking if multiple operator processes ever append concurrently.
    """

    destination = Path(path)
    _check_destination_for_append(destination)
    payload = _render_csv_lines(rows, header=False)
    if not payload:
        return
    try:
        descriptor = os.open(str(destination), os.O_WRONLY | os.O_APPEND)
        try:
            os.write(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise NvmlFailure("atomic_write", str(exc)) from exc


def write_or_append_nvml_csv(path: Path | str, rows: Sequence[Sequence[str]]) -> None:
    """Create ``nvml.csv`` if it does not yet exist, otherwise append to it.

    Convenience wrapper for the common operator loop: the first cell of a run creates the log
    with its header, and every following cell appends. Existence is re-checked immediately
    before acting to keep the create/append decision itself race-narrow; a concurrent creator
    still surfaces as a typed ``output_exists``/``missing_output`` failure rather than silent
    data loss.
    """

    destination = Path(path)
    try:
        destination.lstat()
    except FileNotFoundError:
        write_nvml_csv(destination, rows)
        return
    append_nvml_csv(destination, rows)


__all__ = [
    "NVML_CSV_HEADER",
    "NvmlBackend",
    "NvmlFailure",
    "NvmlSampler",
    "PollWindow",
    "PynvmlBackend",
    "STAGES",
    "append_nvml_csv",
    "write_nvml_csv",
    "write_or_append_nvml_csv",
]
