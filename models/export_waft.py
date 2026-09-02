#!/usr/bin/env python3
"""Export and qualify the pinned WAFT/Twins candidate without network access.

The upstream checkout, checkpoint member, and (optionally) config are supplied by the
operator.  Nothing in this script downloads code, weights, timm initialisation, xformers,
PyTorch, ONNX, or ONNX Runtime.  A candidate ONNX file is published only after strict
checkpoint loading, an ONNX checker/operator-domain gate, PyTorch-vs-ONNX parity, identity,
and both signed translation directions pass.  ``--update-manifest`` records a typed failure
as well as a successful evaluation, but never changes the unresolved checkpoint licence
exclusion or publishes a partial artifact hash.

The large checkpoint and exported ONNX payload are intentionally ignored by git.  The
manifest is an evaluation record, not shipping approval; its checkpoint licence remains
``unknown`` even when all numerical checks pass.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import platform as host_platform
import stat
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

try:
    from artifact_workflow import (  # type: ignore  # pylint: disable=wrong-import-position
        ArtifactError,
        environment_sha256,
        load_manifest,
        publish_file,
        sha256_file,
        write_manifest,
    )
except ModuleNotFoundError:  # pragma: no cover - package import path
    from .artifact_workflow import (  # type: ignore  # pylint: disable=wrong-import-position
        ArtifactError,
        environment_sha256,
        load_manifest,
        publish_file,
        sha256_file,
        write_manifest,
    )


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = SCRIPT_DIR / "waft-twins-artifact.json"
EXPECTED_SOURCE_COMMIT = "b152ff1cad1af8c185ee7b141997c48ff3334c87"
EXPECTED_CHECKPOINT_SHA256 = "f750cd15281fc30de477723438ff4a67fe1591deac4ab0eb9b366e27c827e070"
EXPECTED_CHECKPOINT_SIZE = 544230582
EXPECTED_BACKBONE = "twins"
EXPECTED_CONFIG_PATH = "config/a2/twins/chairs-things.json"
EXPECTED_CONFIG_SHA256 = "4eb827762b132fe0e90b4d87e456088e772573b4f346d5e396e0912dad528996"
OPERATOR_GATE_VERSION = "onnx-standard-domain-v1"
PROVIDER_CHOICES = frozenset(("CPUExecutionProvider", "CUDAExecutionProvider"))


class BlockerCode:
    """Stable technical-blocker vocabulary for an evaluation attempt."""

    MISSING_INPUT = "missing_pinned_input"
    SOURCE_REVISION = "upstream_revision_mismatch"
    SOURCE_DIRTY = "upstream_worktree_dirty"
    CHECKPOINT_IDENTITY = "checkpoint_identity_mismatch"
    CONFIG = "pinned_config_invalid"
    PLATFORM_IDENTITY = "platform_identity_mismatch"
    DEPENDENCY = "missing_export_dependency"
    CHECKPOINT_LOAD = "strict_checkpoint_load_failure"
    ONNX_EXPORT = "onnx_export_failure"
    OPERATOR_DOMAIN = "unsupported_operator_or_domain"
    PARITY = "pytorch_onnx_parity_failure"
    DIRECTION = "direction_or_identity_failure"
    ARTIFACT = "artifact_publication_failure"


@dataclass
class TechnicalBlocker(RuntimeError):
    """A machine-readable, reproducible reason an evaluation did not qualify."""

    code: str
    stage: str
    message: str
    details: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        RuntimeError.__init__(self, self.message)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "code": self.code,
            "stage": self.stage,
            "message": self.message,
            "details": dict(self.details or {}),
        }


def require(condition: bool, message: str, *, code: str, stage: str, details: Mapping[str, Any] | None = None) -> None:
    if not condition:
        raise TechnicalBlocker(code, stage, message, details)


def _regular_file(path: Path, label: str, *, code: str, stage: str) -> None:
    try:
        info = path.lstat()
    except OSError as exc:
        raise TechnicalBlocker(code, stage, f"{label} is missing: {path}") from exc
    if stat.S_ISLNK(info.st_mode):
        raise TechnicalBlocker(code, stage, f"{label} must not be a symlink: {path}")
    if not stat.S_ISREG(info.st_mode):
        raise TechnicalBlocker(code, stage, f"{label} is not a regular file: {path}")


def _git_head(upstream: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(upstream), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise TechnicalBlocker(
            BlockerCode.MISSING_INPUT,
            "provenance",
            f"could not read the pinned upstream checkout: {upstream}",
            {"command": ["git", "-C", str(upstream), "rev-parse", "HEAD"]},
        ) from exc
    return result.stdout.strip()


def _require_clean_worktree(upstream: Path) -> None:
    """Reject local edits/untracked inputs so the commit hash is a complete source identity."""

    try:
        result = subprocess.run(
            ["git", "-C", str(upstream), "status", "--porcelain=v1", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise TechnicalBlocker(
            BlockerCode.MISSING_INPUT,
            "provenance",
            f"could not inspect the upstream worktree: {upstream}",
        ) from exc
    dirty = result.stdout.strip()
    require(
        not dirty,
        "upstream checkout must be clean; refusing a commit hash with local edits or untracked files",
        code=BlockerCode.SOURCE_DIRTY,
        stage="provenance",
        details={"status_porcelain": dirty.splitlines()[:20]},
    )


def verify_provenance(manifest: Mapping[str, Any], upstream: Path, checkpoint: Path) -> dict[str, Any]:
    """Verify only operator-supplied bytes; this function has no network path."""

    git_metadata = upstream / ".git"
    try:
        git_info = git_metadata.lstat()
    except OSError as exc:
        raise TechnicalBlocker(
            BlockerCode.MISSING_INPUT,
            "provenance",
            f"upstream checkout is missing .git metadata: {upstream}",
        ) from exc
    require(
        not stat.S_ISLNK(git_info.st_mode) and (stat.S_ISDIR(git_info.st_mode) or stat.S_ISREG(git_info.st_mode)),
        "upstream .git metadata must be a directory or worktree file",
        code=BlockerCode.MISSING_INPUT,
        stage="provenance",
    )
    actual_commit = _git_head(upstream)
    _require_clean_worktree(upstream)
    expected_commit = manifest["upstream"]["commit"]
    require(
        actual_commit == expected_commit == EXPECTED_SOURCE_COMMIT,
        "upstream checkout is not the manifest's pinned commit",
        code=BlockerCode.SOURCE_REVISION,
        stage="provenance",
        details={"actual": actual_commit, "expected": expected_commit},
    )

    _regular_file(checkpoint, "checkpoint member", code=BlockerCode.MISSING_INPUT, stage="provenance")
    expected_size = manifest["checkpoint"]["size_bytes"]
    actual_size = checkpoint.stat().st_size
    require(
        actual_size == expected_size == EXPECTED_CHECKPOINT_SIZE,
        "checkpoint member size does not match the pinned member",
        code=BlockerCode.CHECKPOINT_IDENTITY,
        stage="provenance",
        details={"actual": actual_size, "expected": expected_size},
    )
    actual_hash = sha256_file(checkpoint)
    expected_hash = manifest["checkpoint"]["sha256"]
    require(
        actual_hash == expected_hash == EXPECTED_CHECKPOINT_SHA256,
        "checkpoint member SHA256 does not match the pinned member",
        code=BlockerCode.CHECKPOINT_IDENTITY,
        stage="provenance",
        details={"actual": actual_hash, "expected": expected_hash},
    )
    return {
        "source_commit": actual_commit,
        "checkpoint_size_bytes": actual_size,
        "checkpoint_sha256": actual_hash,
    }


def _find_config_value(config: Mapping[str, Any], name: str) -> Any:
    """Find a flat config key through the common nested config wrappers."""

    for key, value in config.items():
        if key.lower() == name.lower():
            return value
        if isinstance(value, Mapping):
            found = _find_config_value(value, name)
            if found is not None:
                return found
    return None


def load_pinned_config(manifest: Mapping[str, Any], upstream: Path, path: Path | None) -> tuple[dict[str, Any], Path]:
    configured = path or (upstream / manifest["model"]["config_source"])
    expected_relative = manifest["model"]["config_source"]
    expected_path = upstream / expected_relative
    require(
        expected_relative == EXPECTED_CONFIG_PATH,
        "manifest config path is not the audited WAFT Twins config",
        code=BlockerCode.CONFIG,
        stage="config",
        details={"actual": expected_relative, "expected": EXPECTED_CONFIG_PATH},
    )
    require(
        configured.resolve() == expected_path.resolve(),
        "--config must identify the exact config path inside the pinned upstream checkout",
        code=BlockerCode.CONFIG,
        stage="config",
        details={"actual": str(configured), "expected": str(expected_path)},
    )
    _regular_file(configured, "WAFT config", code=BlockerCode.CONFIG, stage="config")
    expected_hash = manifest["model"].get("config", {}).get("config_sha256")
    require(
        expected_hash == EXPECTED_CONFIG_SHA256,
        "manifest is missing the audited config SHA256",
        code=BlockerCode.CONFIG,
        stage="config",
    )
    actual_hash = sha256_file(configured)
    require(
        actual_hash == expected_hash,
        "pinned WAFT config SHA256 does not match the audited file",
        code=BlockerCode.CONFIG,
        stage="config",
        details={"actual": actual_hash, "expected": expected_hash},
    )
    try:
        value = json.loads(configured.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TechnicalBlocker(
            BlockerCode.CONFIG,
            "config",
            f"could not parse pinned WAFT config: {configured}",
        ) from exc
    require(
        isinstance(value, Mapping),
        "pinned WAFT config must be a JSON object",
        code=BlockerCode.CONFIG,
        stage="config",
    )
    feature_encoder = _find_config_value(value, "feature_encoder")
    require(
        feature_encoder == EXPECTED_BACKBONE,
        "pinned WAFT config is not the Twins variant",
        code=BlockerCode.CONFIG,
        stage="config",
        details={"feature_encoder": feature_encoder},
    )
    require(
        _find_config_value(value, "iterative_module") is not None,
        "pinned WAFT config does not declare iterative_module; refusing to guess it",
        code=BlockerCode.CONFIG,
        stage="config",
    )
    require(
        _find_config_value(value, "iters") is not None,
        "pinned WAFT config does not declare iters; refusing to guess it",
        code=BlockerCode.CONFIG,
        stage="config",
    )
    return dict(value), configured


def _flatten_config(value: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten nested JSON wrappers while retaining the last explicit leaf value."""

    result: dict[str, Any] = {}

    def visit(item: Mapping[str, Any]) -> None:
        for key, nested in item.items():
            if isinstance(nested, Mapping):
                visit(nested)
            else:
                result[str(key)] = nested

    visit(value)
    return result


def construct_weightless_model(
    manifest: Mapping[str, Any],
    upstream: Path,
    config: Mapping[str, Any],
    device: str,
) -> Any:
    """Construct WAFTv2 from the pinned checkout with NO learned weights.

    This is the exact architecture-construction path ``load_model`` runs before it loads the
    checkpoint: it forces timm ``pretrained=False`` and makes any hidden weight download fatal, so
    the returned module carries only randomly-initialised parameters and the checkpoint is the only
    later source of learned tensors.  It is factored out so a checkpoint-free construction smoke
    (CI, or an offline import check) can exercise the real ``from model.waft_a2 import WAFTv2``
    import chain and Twins backbone build without the checkpoint and without drifting from the
    export path.  ``device`` is accepted for signature parity with ``load_model``; the model is
    returned on CPU and the caller performs the checkpoint load and device placement.
    """

    try:
        import torch
        import timm
    except ImportError as exc:
        raise TechnicalBlocker(
            BlockerCode.DEPENDENCY,
            "dependencies",
            "PyTorch and timm are required for WAFT export; install the pinned environment explicitly",
            {"missing": getattr(exc, "name", str(exc))},
        ) from exc

    del device  # architecture-only construction is device-independent; parity with load_model
    flat_config = _flatten_config(config)
    expected_config = manifest["model"].get("config", {})
    args_values = dict(expected_config) if isinstance(expected_config, Mapping) else {}
    args_values.update(flat_config)
    args_values["feature_encoder"] = EXPECTED_BACKBONE
    args_values.setdefault("var_min", -10)
    args_values.setdefault("var_max", 10)
    args = SimpleNamespace(**args_values)
    for field in ("iterative_module", "iters"):
        require(
            hasattr(args, field),
            f"resolved WAFT config does not provide {field}",
            code=BlockerCode.CONFIG,
            stage="config",
        )

    # WAFT's TwinsFeatureEncoder normally asks timm for pretrained weights.  Force the
    # constructor to use an uninitialised architecture and make any hidden hub path fatal;
    # the exact checkpoint must be the only source of learned tensors.
    original_create_model = timm.create_model
    original_hub_loader = getattr(torch.hub, "load_state_dict_from_url", None)
    patched_create_model_targets: list[tuple[Any, str, Any]] = []

    def no_pretrained_create_model(name: str, *positional: Any, **kwargs: Any) -> Any:
        kwargs["pretrained"] = False
        return original_create_model(name, *positional, **kwargs)

    def forbidden_hub_loader(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("unexpected network weight download while constructing WAFT")

    sys.path.insert(0, str(upstream))
    try:
        try:
            from model.waft_a2 import WAFTv2  # type: ignore

            timm.create_model = no_pretrained_create_model
            # Some pinned timm integrations import ``create_model`` into the backbone module
            # rather than looking it up through ``timm``.  Patch both spellings so either
            # implementation remains download-free, then restore the module exactly.
            twins_module = sys.modules.get("model.backbone.twins")
            if twins_module is not None:
                for attribute in ("create_model",):
                    if hasattr(twins_module, attribute):
                        original = getattr(twins_module, attribute)
                        setattr(twins_module, attribute, no_pretrained_create_model)
                        patched_create_model_targets.append((twins_module, attribute, original))
            if original_hub_loader is not None:
                torch.hub.load_state_dict_from_url = forbidden_hub_loader
            model = WAFTv2(args)
        except TechnicalBlocker:
            raise
        except Exception as exc:
            raise TechnicalBlocker(
                BlockerCode.CHECKPOINT_LOAD,
                "model_construction",
                "could not construct WAFTv2 from the pinned checkout without downloading weights",
                {"exception": type(exc).__name__, "message": str(exc)},
            ) from exc
    finally:
        for target, attribute, original in patched_create_model_targets:
            setattr(target, attribute, original)
        timm.create_model = original_create_model
        if original_hub_loader is not None:
            torch.hub.load_state_dict_from_url = original_hub_loader
        try:
            sys.path.remove(str(upstream))
        except ValueError:  # pragma: no cover - defensive cleanup
            pass
    return model


def load_model(
    manifest: Mapping[str, Any],
    upstream: Path,
    checkpoint: Path,
    config: Mapping[str, Any],
    device: str,
) -> tuple[Any, dict[str, Any]]:
    """Construct the weightless WAFT architecture, then strictly load the local checkpoint."""

    model = construct_weightless_model(manifest, upstream, config, device)
    # construct_weightless_model raises the DEPENDENCY blocker if torch is unavailable, so by here
    # the import is guaranteed to succeed (and is cached).
    import torch

    try:
        # ``weights_only=True`` is intentional: accepting arbitrary pickle code would make
        # the user-supplied checkpoint an executable input.  A wrapper under ``model`` is
        # accepted because WAFT's evaluator emits that shape; no key is renamed or dropped.
        state = torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise TechnicalBlocker(
            BlockerCode.CHECKPOINT_LOAD,
            "checkpoint_load",
            "PyTorch could not safely read the local checkpoint with weights_only=True",
            {"exception": type(exc).__name__, "message": str(exc)},
        ) from exc
    if isinstance(state, Mapping) and isinstance(state.get("model"), Mapping):
        state = state["model"]
    require(
        isinstance(state, Mapping),
        "checkpoint did not contain a state-dict mapping",
        code=BlockerCode.CHECKPOINT_LOAD,
        stage="checkpoint_load",
    )
    try:
        model.load_state_dict(state, strict=True)
    except Exception as exc:
        raise TechnicalBlocker(
            BlockerCode.CHECKPOINT_LOAD,
            "checkpoint_load",
            "WAFT checkpoint failed strict=True state-dict loading",
            {
                "exception": type(exc).__name__,
                "message": str(exc),
                "state_dict_keys": len(state),
            },
        ) from exc
    try:
        model.eval().to(torch.device(device))
    except Exception as exc:
        raise TechnicalBlocker(
            BlockerCode.CHECKPOINT_LOAD,
            "device_setup",
            f"could not place the strictly loaded WAFT model on {device}",
            {"exception": type(exc).__name__, "message": str(exc), "device": device},
        ) from exc
    return model, {
        "strict": True,
        "state_dict_keys": len(state),
        "missing_keys": 0,
        "unexpected_keys": 0,
        "pretrained_initialization": False,
        "pytorch": str(torch.__version__),
    }


def _flow_wrapper(torch: Any, model: Any) -> Any:
    class FinalFlow(torch.nn.Module):
        def __init__(self, wrapped: Any) -> None:
            super().__init__()
            self.wrapped = wrapped

        def forward(self, image1: Any, image2: Any) -> Any:
            result = self.wrapped(image1, image2)
            if not isinstance(result, Mapping) or "flow" not in result:
                raise RuntimeError("WAFT forward did not return a flow mapping")
            flows = result["flow"]
            if not isinstance(flows, (list, tuple)) or not flows:
                raise RuntimeError("WAFT forward returned no flow predictions")
            return flows[-1]

    return FinalFlow(model)


def synthetic_pair(torch: Any, height: int, width: int, dx: int, seed: int, device: str) -> tuple[Any, Any]:
    require(
        dx > 0 and width > dx,
        "synthetic translation must be positive and fit inside the test width",
        code=BlockerCode.DIRECTION,
        stage="synthetic_validation",
    )
    generator = torch.Generator(device=device).manual_seed(seed)
    first = torch.rand((1, 3, height, width), generator=generator, device=device) * 255.0
    first = torch.nn.functional.avg_pool2d(first, kernel_size=5, stride=1, padding=2)
    second = torch.zeros_like(first)
    second[..., dx:] = first[..., :-dx]
    return first, second


def validate_provider_device(device: str, provider: str) -> None:
    require(
        provider in PROVIDER_CHOICES,
        "requested ONNX Runtime provider is not in the qualification matrix",
        code=BlockerCode.CONFIG,
        stage="provider_setup",
        details={"provider": provider, "allowed": sorted(PROVIDER_CHOICES)},
    )
    require(
        (provider == "CUDAExecutionProvider") == (device == "cuda"),
        "PyTorch device and ONNX Runtime provider must be selected as the same CPU/CUDA path",
        code=BlockerCode.CONFIG,
        stage="provider_setup",
        details={"device": device, "provider": provider},
    )


def verify_provider_selection(requested: str, actual: Sequence[str]) -> None:
    require(
        actual and actual[0] == requested,
        "ONNX Runtime did not activate the explicitly requested provider",
        code=BlockerCode.DEPENDENCY,
        stage="provider_setup",
        details={"requested": requested, "actual": list(actual)},
    )


def _opset_versions(graph: Any) -> dict[str, int]:
    values: dict[str, int] = {}
    for item in getattr(graph, "opset_import", ()):  # pragma: no branch - protobuf iterable
        domain = getattr(item, "domain", "") or "ai.onnx"
        values[domain] = int(getattr(item, "version", 0))
    return values


def gate_onnx_graph(model_proto: Any, *, expected_opset: int) -> dict[str, Any]:
    """Reject custom domains and operators absent from the declared standard ONNX opset."""

    graph = model_proto.graph
    nodes = list(getattr(graph, "node", ()))
    domains = sorted({getattr(node, "domain", "") or "ai.onnx" for node in nodes})
    ops = sorted({str(getattr(node, "op_type", "")) for node in nodes})
    opsets = _opset_versions(graph)
    foreign_domains = sorted(set(domains) - {"ai.onnx"})
    if "ai.onnx" in opsets and opsets["ai.onnx"] != expected_opset:
        raise TechnicalBlocker(
            BlockerCode.OPERATOR_DOMAIN,
            "onnx_gate",
            "exported graph has an unexpected ai.onnx opset",
            {"actual": opsets["ai.onnx"], "expected": expected_opset},
        )
    if foreign_domains:
        raise TechnicalBlocker(
            BlockerCode.OPERATOR_DOMAIN,
            "onnx_gate",
            "exported graph contains a non-standard ONNX operator domain",
            {"domains": domains, "foreign_domains": foreign_domains},
        )

    unsupported: list[str] = []
    try:
        import onnx

        for op in ops:
            try:
                onnx.defs.get_schema(op, max_inclusive_version=expected_opset)
            except Exception:
                unsupported.append(op)
    except ImportError:  # pragma: no cover - export itself already imports onnx
        unsupported = []
    if unsupported:
        raise TechnicalBlocker(
            BlockerCode.OPERATOR_DOMAIN,
            "onnx_gate",
            "exported graph contains operators unavailable in the declared standard opset",
            {"unsupported_operators": unsupported, "opset": expected_opset},
        )
    return {
        "version": OPERATOR_GATE_VERSION,
        "allowed_domains": ["ai.onnx"],
        "domains": domains,
        "foreign_domains": foreign_domains,
        "opsets": opsets,
        "operators": ops,
        "unsupported_operators": unsupported,
    }


def export_onnx(model: Any, manifest: Mapping[str, Any], output: Path, device: str) -> dict[str, Any]:
    try:
        import onnx
        import torch
    except ImportError as exc:
        raise TechnicalBlocker(
            BlockerCode.DEPENDENCY,
            "dependencies",
            "ONNX and PyTorch are required for WAFT export",
            {"missing": getattr(exc, "name", str(exc))},
        ) from exc

    shape = manifest["export"]["example_shape"]
    require(
        shape[0] == 1 and shape[1] == 3 and shape[2] % 32 == 0 and shape[3] % 32 == 0,
        "WAFT example shape must be N=1, RGB, and a multiple of 32",
        code=BlockerCode.CONFIG,
        stage="onnx_export",
        details={"shape": shape},
    )
    wrapper = _flow_wrapper(torch, model).eval()
    sample1 = torch.zeros(shape, dtype=torch.float32, device=device)
    sample2 = torch.zeros(shape, dtype=torch.float32, device=device)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        torch.onnx.export(
            wrapper,
            (sample1, sample2),
            str(output),
            export_params=True,
            opset_version=manifest["export"]["opset"],
            do_constant_folding=True,
            input_names=[item["name"] for item in manifest["tensor_contract"]["inputs"]],
            output_names=[manifest["tensor_contract"]["output"]["name"]],
            dynamic_axes={
                "image1": {2: "height", 3: "width"},
                "image2": {2: "height", 3: "width"},
                "flow": {2: "height", 3: "width"},
            },
        )
        exported = onnx.load(str(output), load_external_data=False)
        operator_gate = gate_onnx_graph(exported, expected_opset=manifest["export"]["opset"])
        onnx.checker.check_model(exported, full_check=True)
        return operator_gate
    except TechnicalBlocker:
        raise
    except Exception as exc:
        raise TechnicalBlocker(
            BlockerCode.ONNX_EXPORT,
            "onnx_export",
            "PyTorch could not export WAFT to a checked ONNX graph",
            {"exception": type(exc).__name__, "message": str(exc)},
        ) from exc


def validate_export(
    model: Any,
    manifest: Mapping[str, Any],
    output: Path,
    device: str,
    strict_load: Mapping[str, Any],
    operator_gate: Mapping[str, Any],
    config_path: Path,
    provider: str,
) -> dict[str, Any]:
    try:
        import numpy as np
        import onnx
        import onnxruntime as ort
        import torch
    except ImportError as exc:
        raise TechnicalBlocker(
            BlockerCode.DEPENDENCY,
            "dependencies",
            "NumPy, ONNX, and ONNX Runtime are required for WAFT qualification",
            {"missing": getattr(exc, "name", str(exc))},
        ) from exc

    shape = manifest["export"]["example_shape"]
    validation = manifest["validation"]
    dx = int(validation["translation_pixels"])
    first, second = synthetic_pair(torch, shape[2], shape[3], dx, validation["seed"], device)
    wrapper = _flow_wrapper(torch, model).eval()

    def run_pt(a: Any, b: Any) -> Any:
        with torch.no_grad():
            value = wrapper(a, b)
        return value.detach().cpu().numpy()

    identity_pt = run_pt(first, first)
    forward_pt = run_pt(first, second)
    reverse_pt = run_pt(second, first)
    validate_provider_device(device, provider)
    session = ort.InferenceSession(str(output), providers=[provider])
    actual_providers = session.get_providers()
    verify_provider_selection(provider, actual_providers)
    input_names = [item["name"] for item in manifest["tensor_contract"]["inputs"]]
    output_name = manifest["tensor_contract"]["output"]["name"]
    actual_inputs = [item.name for item in session.get_inputs()]
    require(
        actual_inputs == input_names,
        "ONNX input names do not match the frozen tensor contract",
        code=BlockerCode.ONNX_EXPORT,
        stage="onnx_validation",
        details={"actual": actual_inputs, "expected": input_names},
    )

    def run_onnx(a: Any, b: Any) -> Any:
        return session.run(
            [output_name],
            {input_names[0]: a.detach().cpu().numpy(), input_names[1]: b.detach().cpu().numpy()},
        )[0]

    identity_onnx = run_onnx(first, first)
    forward_onnx = run_onnx(first, second)
    reverse_onnx = run_onnx(second, first)
    second_shape = validation["second_dynamic_shape"]
    require(
        second_shape[0] == 1 and second_shape[1] == 3 and second_shape[2] % 32 == 0 and second_shape[3] % 32 == 0,
        "second WAFT validation shape must be a multiple of 32",
        code=BlockerCode.CONFIG,
        stage="onnx_validation",
        details={"shape": second_shape},
    )
    dynamic_first, dynamic_second = synthetic_pair(
        torch, second_shape[2], second_shape[3], dx, validation["seed"] + 1, device
    )
    dynamic_pt = run_pt(dynamic_first, dynamic_second)
    dynamic_onnx = run_onnx(dynamic_first, dynamic_second)
    expected_dynamic_shape = [1, 2, second_shape[2], second_shape[3]]
    require(
        list(dynamic_onnx.shape) == expected_dynamic_shape,
        "ONNX graph did not preserve dynamic spatial dimensions",
        code=BlockerCode.ONNX_EXPORT,
        stage="onnx_validation",
        details={"actual": list(dynamic_onnx.shape), "expected": expected_dynamic_shape},
    )

    pairs = (
        ("identity", identity_pt, identity_onnx),
        ("forward", forward_pt, forward_onnx),
        ("reverse", reverse_pt, reverse_onnx),
        ("second_shape", dynamic_pt, dynamic_onnx),
    )
    for label, pytorch_value, onnx_value in pairs:
        require(
            np.all(np.isfinite(onnx_value)) and np.all(np.isfinite(pytorch_value)),
            f"{label} produced non-finite flow values",
            code=BlockerCode.PARITY,
            stage="numerical_validation",
            details={"case": label},
        )
    differences = np.concatenate([(pt - exported).reshape(-1) for _, pt, exported in pairs])
    absolute = np.abs(differences)
    mean_abs = float(np.mean(absolute))
    p99_abs = float(np.percentile(absolute, 99.0))
    p999_abs = float(np.percentile(absolute, 99.9))
    max_abs = float(np.max(absolute))
    require(
        mean_abs <= validation["onnx_pytorch_mean_abs_max"]
        and p99_abs <= validation["onnx_pytorch_p99_abs_max"]
        and p999_abs <= validation["onnx_pytorch_p999_abs_max"]
        and max_abs <= validation["onnx_pytorch_max_abs_max"],
        "PyTorch-vs-ONNX parity exceeds a frozen threshold",
        code=BlockerCode.PARITY,
        stage="numerical_validation",
        details={
            "mean_abs": mean_abs,
            "p99_abs": p99_abs,
            "p999_abs": p999_abs,
            "max_abs": max_abs,
        },
    )

    border = max(8, dx * 2)
    if min(shape[2], shape[3]) <= border * 2:
        border = max(1, min(shape[2], shape[3]) // 8)
    interior = np.s_[0, :, border:-border, border:-border]
    identity_epe = np.sqrt(np.sum(identity_onnx[interior] ** 2, axis=0))
    identity_median = float(np.median(identity_epe))
    forward_x = float(np.median(forward_onnx[0, 0, border:-border, border:-border]))
    forward_y = float(np.median(forward_onnx[0, 1, border:-border, border:-border]))
    reverse_x = float(np.median(reverse_onnx[0, 0, border:-border, border:-border]))
    reverse_y = float(np.median(reverse_onnx[0, 1, border:-border, border:-border]))
    minimum_x = dx * validation["translation_x_fraction_min"]
    require(
        identity_median <= validation["identity_median_epe_max"]
        and forward_x >= minimum_x
        and reverse_x <= -minimum_x
        and abs(forward_y) <= validation["translation_abs_y_max"]
        and abs(reverse_y) <= validation["translation_abs_y_max"],
        "identity or signed translation direction check failed",
        code=BlockerCode.DIRECTION,
        stage="numerical_validation",
        details={
            "identity_median_epe": identity_median,
            "forward_median": [forward_x, forward_y],
            "reverse_median": [reverse_x, reverse_y],
            "minimum_x": minimum_x,
        },
    )
    graph = onnx.load(str(output), load_external_data=False)
    return {
        "environment": {
            "platform": host_platform.platform(),
            "python": host_platform.python_version(),
            "pytorch": str(torch.__version__),
            "onnx": str(onnx.__version__),
            "onnxruntime": str(ort.__version__),
            "requested_provider": provider,
            "providers": actual_providers,
        },
        "config": {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
        },
        "strict_checkpoint": dict(strict_load),
        "graph_nodes": len(graph.graph.node),
        "operator_gate": dict(operator_gate),
        "onnx_pytorch_mean_abs": mean_abs,
        "onnx_pytorch_p99_abs": p99_abs,
        "onnx_pytorch_p999_abs": p999_abs,
        "onnx_pytorch_max_abs": max_abs,
        "identity_median_epe": identity_median,
        "forward_median": [forward_x, forward_y],
        "reverse_median": [reverse_x, reverse_y],
        "second_dynamic_shape": list(dynamic_onnx.shape),
    }


def _detected_platform_id() -> str:
    if sys.platform == "darwin":
        return "macos-arm64" if host_platform.machine().lower() in {"arm64", "aarch64"} else "macos-x86_64"
    if sys.platform.startswith("linux"):
        return f"linux-{host_platform.machine().lower()}"
    return f"{sys.platform}-{host_platform.machine().lower()}"


def _platform_id(value: str | None) -> str:
    detected = _detected_platform_id()
    selected = value or detected
    require(
        selected == detected,
        "--platform must match the platform and architecture that produced this export",
        code=BlockerCode.PLATFORM_IDENTITY,
        stage="platform",
        details={"requested": selected, "detected": detected},
    )
    return selected


def _platform_parts(platform_id: str) -> tuple[str, str]:
    if platform_id.startswith("macos-"):
        return "macos", platform_id.removeprefix("macos-")
    if platform_id.startswith("linux-"):
        return "linux", platform_id.removeprefix("linux-")
    return platform_id.split("-", 1)[0], platform_id.split("-", 1)[-1]


def _environment(observed: Mapping[str, Any], platform_id: str) -> dict[str, Any]:
    environment_platform, environment_architecture = _platform_parts(platform_id)
    return {
        "platform": environment_platform,
        "architecture": environment_architecture,
        "python": observed["environment"]["python"],
        "framework": f"pytorch=={observed['environment']['pytorch']}",
        "exporter": f"onnx=={observed['environment']['onnx']}",
        "runtime": f"onnxruntime=={observed['environment']['onnxruntime']}",
        "provider": ",".join(observed["environment"].get("providers", ["CPUExecutionProvider"])),
    }


def update_success(
    manifest_path: Path,
    manifest: dict[str, Any],
    output: Path,
    observed: Mapping[str, Any],
    platform_id: str,
) -> None:
    """Atomically record an exact evaluation artifact while retaining status=excluded."""

    env = _environment(observed, platform_id)
    env["sha256"] = environment_sha256(env)
    digest = sha256_file(output)
    size = output.stat().st_size
    entry = {
        "platform": platform_id,
        "artifact": output.name,
        "sha256": digest,
        "size_bytes": size,
        "mode": "0644",
        "export_environment_sha256": env["sha256"],
        "export_environment": env,
    }
    export = manifest["export"]
    export["platform"] = platform_id
    export["artifact"] = output.name
    export["sha256"] = digest
    export["size_bytes"] = size
    export["mode"] = "0644"
    export["export_environment_sha256"] = env["sha256"]
    export["platform_artifacts"] = [entry]
    manifest["export_environment"] = env
    validation = manifest["validation"]
    validation["status"] = "passed"
    validation["identity"] = {"passed": True, "median_epe_px": observed["identity_median_epe"]}
    validation["directions"] = {
        "forward": {
            "median_dx_px": observed["forward_median"][0],
            "median_dy_px": observed["forward_median"][1],
            "expected_sign": "positive_x",
        },
        "reverse": {
            "median_dx_px": observed["reverse_median"][0],
            "median_dy_px": observed["reverse_median"][1],
            "expected_sign": "negative_x",
        },
    }
    shape = manifest["export"]["example_shape"]
    validation["shapes"] = {
        "dynamic": True,
        "example": [1, 2, shape[2], shape[3]],
        "additional": observed["second_dynamic_shape"],
    }
    validation["parity"] = {
        "checked": True,
        "mean_abs": observed["onnx_pytorch_mean_abs"],
        "p99_abs": observed["onnx_pytorch_p99_abs"],
        "p999_abs": observed["onnx_pytorch_p999_abs"],
        "max_abs": observed["onnx_pytorch_max_abs"],
    }
    validation["observed"] = dict(observed)
    # Do not set status to export_validated/host_probe_pending: unresolved checkpoint terms
    # intentionally keep this evaluation artifact excluded from all shipping paths.
    manifest["status"] = "excluded"
    write_manifest(manifest_path, manifest)


def record_failure(manifest_path: Path, manifest: dict[str, Any], blocker: TechnicalBlocker) -> bool:
    """Record a blocker as pending evidence; never erase a successful artifact claim.

    The shared artifact contract reserves ``validation.status=failed`` for an evaluated
    artifact result.  A missing local input or unavailable provider is still pending, so its
    typed blocker is stored in ``observed`` without pretending a model artifact was tested.
    """

    export = manifest["export"]
    if export.get("sha256") is not None or export.get("size_bytes") is not None:
        return False
    if manifest["validation"].get("status") == "passed":
        return False
    manifest["status"] = "excluded"
    manifest["validation"]["status"] = "pending"
    manifest["validation"]["observed"] = {"technical_blocker": blocker.as_dict()}
    write_manifest(manifest_path, manifest)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True, type=Path, help="local checkout at the manifest's exact WAFT commit")
    parser.add_argument("--checkpoint", required=True, type=Path, help="local extracted zero-shot.pth member with the manifest's exact hash")
    parser.add_argument("--config", type=Path, help="local config JSON; defaults to the manifest path inside --upstream")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, help="staged ONNX path; defaults beside the manifest")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--provider",
        choices=sorted(PROVIDER_CHOICES),
        default="CPUExecutionProvider",
        help="explicit ONNX Runtime execution provider; no fallback is accepted",
    )
    parser.add_argument("--platform", help="platform identity for this exact export")
    parser.add_argument("--verify-provenance-only", action="store_true", help="verify only source/checkpoint bytes")
    parser.add_argument("--update-manifest", action="store_true", help="record a successful artifact or a typed pending blocker")
    parser.add_argument("--record-failure", action="store_true", help="record a typed blocker without requiring a successful export")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    manifest = load_manifest(manifest_path)
    upstream = args.upstream.resolve()
    checkpoint = args.checkpoint.resolve()
    try:
        provenance = verify_provenance(manifest, upstream, checkpoint)
        print(f"provenance: source {provenance['source_commit']} and checkpoint {provenance['checkpoint_sha256']} verified")
        if args.verify_provenance_only:
            return 0
        validate_provider_device(args.device, args.provider)
        config, config_path = load_pinned_config(manifest, upstream, args.config.resolve() if args.config else None)
        model, strict_load = load_model(manifest, upstream, checkpoint, config, args.device)
        output = (args.output or (manifest_path.parent / manifest["export"]["artifact"])).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(prefix=output.name + ".candidate.", suffix=".onnx", dir=output.parent, delete=False) as stream:
            candidate = Path(stream.name)
        candidate.unlink()
        try:
            operator_gate = export_onnx(model, manifest, candidate, args.device)
            observed = validate_export(
                model,
                manifest,
                candidate,
                args.device,
                strict_load,
                operator_gate,
                config_path,
                args.provider,
            )
            try:
                publish_file(candidate, output)
            except (ArtifactError, OSError) as exc:
                raise TechnicalBlocker(
                    BlockerCode.ARTIFACT,
                    "artifact_publication",
                    "could not atomically publish the qualified ONNX artifact",
                    {"exception": type(exc).__name__, "message": str(exc), "output": str(output)},
                ) from exc
        except TechnicalBlocker:
            candidate.unlink(missing_ok=True)
            raise
        except OSError as exc:
            candidate.unlink(missing_ok=True)
            raise TechnicalBlocker(BlockerCode.ARTIFACT, "artifact_publication", str(exc)) from exc
        artifact_hash = sha256_file(output)
        print(f"artifact: {output}")
        print(f"sha256:   {artifact_hash}")
        print(f"validation: {json.dumps(observed, sort_keys=True)}")
        if args.update_manifest:
            update_success(manifest_path, manifest, output, observed, _platform_id(args.platform))
            print(f"manifest updated: {manifest_path}")
        return 0
    except TechnicalBlocker as blocker:
        recorded = False
        if args.update_manifest or args.record_failure:
            try:
                recorded = record_failure(manifest_path, manifest, blocker)
            except (ArtifactError, OSError, KeyError, TypeError, ValueError) as exc:
                print(f"manifest failure record unavailable: {exc}", file=sys.stderr)
        payload = json.dumps(blocker.as_dict(), sort_keys=True)
        print(f"technical blocker: {payload}", file=sys.stderr)
        if recorded:
            print(f"manifest blocker recorded: {manifest_path}", file=sys.stderr)
        return 2
    except (ArtifactError, OSError, KeyError, TypeError, ValueError) as exc:
        print(f"export_waft.py: error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
