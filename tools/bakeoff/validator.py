#!/usr/bin/env python3
"""Small, dependency-free JSON Schema and Phase 2.5 protocol validator.

The bake-off runs on an air-gapped EL8 machine.  Pulling in ``jsonschema`` (or any
other package) for the report gate would make the evaluator depend on an unrecorded
Python installation, so this module implements the intentionally small JSON Schema
subset used by the checked-in v1 schemas and adds the cross-document rules that JSON
Schema cannot express (token matrices, repetition counts, and coverage).
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import math
import re
from itertools import product
from pathlib import Path
from typing import Any, Iterable, Mapping, NamedTuple, Sequence


class ValidationError(ValueError):
    """A stable, user-facing validation error with a JSON path."""

    def __init__(self, path: str, message: str):
        self.path = path
        self.message = message
        super().__init__(f"{path}: {message}")


def _load_sibling(module_name: str):
    """Load a bake-off sibling in package, script, and private-spec contexts.

    The artifact workflow loads this file through ``spec_from_file_location`` under a private
    name, where relative imports have no package and ordinary imports cannot see this directory.
    Loading by the sibling path keeps that path dependency-free without mutating ``sys.path``.
    """

    if __package__:
        return importlib.import_module(f".{module_name}", __package__)
    sibling_path = Path(__file__).resolve().with_name(f"{module_name}.py")
    private_name = f"_whitewater_p25_bakeoff_{module_name}"
    spec = importlib.util.spec_from_file_location(private_name, sibling_path)
    if spec is None or spec.loader is None:  # pragma: no cover - checked-in siblings exist
        raise ImportError(f"could not load bake-off sibling: {sibling_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


geometry = _load_sibling("geometry")
metrics = _load_sibling("metrics")


def canonical_sha256(value: Any) -> str:
    """Return the protocol's deterministic SHA256 for a JSON value.

    Matrix selectors are hashed after removing their ``matrix_sha256`` member.  Keeping
    this small canonicalization helper public lets the fixture driver construct mutated
    selectors without duplicating the report contract in test code.
    """

    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


_EXPECTED_CANDIDATES = [
    {"id": "sea-raft-m", "role": "shipping-candidate"},
    {"id": "waft-twins", "role": "shipping-candidate"},
    {"id": "neuflow-v2", "role": "shipping-candidate"},
    {"id": "raft-original", "role": "validation-baseline"},
]
_V1_PROTOCOL_ID = "whitewater-p25-v1"
_V2_PROTOCOL_ID = "whitewater-p25-v2"
_EXPECTED_MEASUREMENT_STATUSES = ["measurable", "unavailable"]
_EXPECTED_REQUIRED_IDENTITY = [
    "source_commit", "checkpoint_sha256", "artifact_sha256", "export_environment_sha256",
]
_EXPECTED_LICENSE_SURFACES = ["code", "checkpoint", "backbone"]
_EXPECTED_REQUIRED_LICENSE_VERDICTS = ["commercial_use_permitted", "redistribution_permitted"]
_EXPECTED_TENSOR_CONTRACT = {
    "batch": 1,
    "input_dtype": "float32",
    "input_layout": "NCHW",
    "input_channels": "RGB",
    "input_pair": ["image1", "image2"],
    "output_dtype": "float32",
    "output_layout": "NCHW",
    "output_channels": ["dx", "dy"],
    "output_direction": "image1_to_image2",
    "output_units": "unpadded_analysis_pixels",
}
_EXPECTED_PROVIDERS_V1 = [
    {"token": "cpu", "environment": "el8-x86_64", "purpose": "correctness", "cap_tokens": ["mp0_5"]},
    {"token": "cuda", "environment": "el8-x86_64", "purpose": "selection",
     "cap_tokens": ["mp0_5", "mp1", "mp2", "mp4", "mp6", "mp8"]},
    {"token": "coreml", "environment": "macos-arm64", "purpose": "supporting",
     "cap_tokens": ["mp0_5", "mp1", "mp2"]},
]
_EXPECTED_PROVIDERS_V2 = [
    {"token": "cpu", "environment": "el8-x86_64", "purpose": "correctness", "cap_tokens": ["mp0_5", "mp0_331776"]},
    {"token": "cuda", "environment": "el8-x86_64", "purpose": "selection",
     "cap_tokens": ["mp0_5", "mp1", "mp2", "mp4", "mp6", "mp8", "mp0_331776"]},
    {"token": "coreml", "environment": "macos-arm64", "purpose": "supporting",
     "cap_tokens": ["mp0_5", "mp1", "mp2", "mp0_331776"]},
]
_EXPECTED_CAPS_V1 = [
    {"token": "mp0_5", "decimal_megapixels": 0.5},
    {"token": "mp1", "decimal_megapixels": 1.0},
    {"token": "mp2", "decimal_megapixels": 2.0},
    {"token": "mp4", "decimal_megapixels": 4.0},
    {"token": "mp6", "decimal_megapixels": 6.0},
    {"token": "mp8", "decimal_megapixels": 8.0},
]
_EXPECTED_CAPS_V2 = [
    *_EXPECTED_CAPS_V1,
    {
        "token": "mp0_331776",
        "decimal_megapixels": 0.331776,
        "lattice": {
            "analysis_width": 768,
            "analysis_height": 432,
            "canonical_aspect_ratio": "16:9",
        },
    },
]
_EXPECTED_CANDIDATE_CONSTRAINTS_V2 = [
    {
        "candidate_id": "neuflow-v2",
        "providers": ["cpu", "cuda"],
        "cap_tokens": ["mp0_331776"],
        "required_geometry": {
            "analysis_width": 768,
            "analysis_height": 432,
            "canonical_aspect_ratio": "16:9",
        },
    },
]
_EXPECTED_CAP_ACCOUNTING = {
    "unit_pixels": 1000000,
    "applies_to": "unpadded_square_pixel_analysis_area",
    "rounding": "phase2-analysisGeometry-v1",
    "record_padded_dimensions": True,
    "minimum_target_cap_token": "mp2",
    "source_targets": ["fhd-1920x1080-par1", "uhd-3840x2160-par1"],
}
_EXPECTED_CONDITIONING = [
    {"token": "native-clamp01-v1", "accepted_encoding": "scene-linear-or-log",
     "formula": "c(x)=min(1,max(0,x)); then artifact-declared packing"},
    {"token": "signed-log-v1", "accepted_encoding": "scene-linear",
     "formula": "c(x)=clamp(0.5+sign(x)*log1p(abs(x))/(2*log(17)),0,1); sign(0)=0; nonfinite input -> typed conditioning failure; then artifact-declared packing"},
    {"token": "pair-percentile-v1", "accepted_encoding": "scene-linear-or-log",
     "formula": "finite RGB samples from both frames; lo=linear-quantile(p=0.01), hi=linear-quantile(p=0.99); c(x)=min(1,max(0,(x-lo)/max(hi-lo,1e-6))); then artifact-declared packing"},
    {"token": "native-log-v1", "accepted_encoding": "log",
     "formula": "c(x)=x unchanged; then artifact-declared packing"},
]
_EXPECTED_QUANTILE = {
    "ordering": "ascending IEEE-754 numeric order after discarding nonfinite samples",
    "index": "h=(n-1)*p",
    "interpolation": "x[floor(h)]+(h-floor(h))*(x[ceil(h)]-x[floor(h)])",
    "scope": "all RGB channel samples from the two-frame pair",
    "empty_pair_result": "typed conditioning failure",
}
_EXPECTED_PADDING_COMPARISON_POLICY = (
    "artifact-declared caller padding, held identical across artifacts in a comparison cell"
)
_EXPECTED_SYNTHETIC_CATEGORIES = [
    "identity", "translation-x", "translation-y", "affine-spatial", "border",
    "occlusion-reveal", "blur", "noise", "hdr-log", "odd-padding", "par", "chain",
]
_EXPECTED_REQUIRED_CASES = [
    "identity",
    "translation-x-positive", "translation-x-negative",
    "translation-y-positive", "translation-y-negative",
    "affine", "spatial", "border", "occlusion-reveal", "blur", "noise",
    "hdr-scene-linear", "log-input", "odd-size", "asymmetric-padding",
    "par-0_5", "par-2", "chain-1", "chain-2", "chain-4", "chain-8",
    "fhd-1920x1080-par1", "uhd-3840x2160-par1",
]
_EXPECTED_PRODUCTION_CATEGORIES = [
    "motion-blur", "defocus", "low-contrast", "grain", "occlusion-reveal",
    "rolling-shutter", "fine-detail", "reflections-screens", "anamorphic",
]
_EXPECTED_METRICS = [
    "endpoint_error_px", "fraction_le_1px", "fraction_le_3px", "landmark_median_error_px",
    "landmark_p95_error_px", "visible_warp_residual", "forward_backward_residual_px",
    "chain_drift_px", "nonfinite_fraction", "repeated_run_p99_delta_px",
]
_EXPECTED_REVIEW_DIMENSIONS = ["edge_adherence", "occlusion_reveal", "blur", "jitter", "drift"]
_EXPECTED_AGGREGATION = {
    "order": ["samples_to_shot", "shots_to_category", "categories_to_partition"],
    "sample_statistic": "median; p95 also reported",
    "shot_weighting": "equal_within_category",
    "category_weighting": "equal",
    "dense_score": "100*(0.50*fraction_le_1px+0.30*fraction_le_3px+0.20*max(0,1-endpoint_error_px/3))",
    "review_score": "100*mean(five_review_dimensions)/4",
    "final_quality_score": "0.30*synthetic_macro_score+0.70*production_macro_score",
}
_EXPECTED_HARD_GATES = {
    "missing_required_results_max": 0,
    "nonfinite_fraction_max": 0.0,
    "repeated_run_p99_delta_px_max": 0.05,
    "peak_incremental_device_memory_gib_max": 15.0,
    "synthetic_macro_score_min": 75.0,
    "synthetic_primary_category_score_min": 60.0,
    "production_macro_score_min": 70.0,
    "production_primary_category_score_min": 60.0,
    "mandatory_source_cap_cells": [
        {"source_target": "fhd-1920x1080-par1", "cap_token": "mp2", "provider": "cuda"},
        {"source_target": "uhd-3840x2160-par1", "cap_token": "mp2", "provider": "cuda"},
    ],
}


class PerCellHardGate(NamedTuple):
    """One hard gate that applies to a completed per-cell measurement."""

    protocol_key: str
    result_section: str
    result_key: str
    validation_message: str
    failure_stage: str


# Keep the executable per-cell gate contract next to the validator's protocol expectations.  The
# runner imports this public specification so a new per-cell gate is classified before report
# publication and validated with the same result path/message without a second hand-maintained
# list.
PER_CELL_HARD_GATES: tuple[PerCellHardGate, ...] = (
    PerCellHardGate(
        protocol_key="nonfinite_fraction_max",
        result_section="metrics",
        result_key="nonfinite_fraction",
        validation_message="pass result exceeds nonfinite gate",
        failure_stage="metrics",
    ),
    PerCellHardGate(
        protocol_key="repeated_run_p99_delta_px_max",
        result_section="metrics",
        result_key="repeated_run_p99_delta_px",
        validation_message="pass result exceeds repeated-run gate",
        failure_stage="metrics",
    ),
    PerCellHardGate(
        protocol_key="peak_incremental_device_memory_gib_max",
        result_section="resource",
        result_key="peak_incremental_device_memory_gib",
        validation_message="pass result exceeds memory gate",
        failure_stage="resource",
    ),
)
_EXPECTED_RANKING = {
    "material_quality_points": 3.0,
    "default_order": [
        "quality_score_desc", "geomean_steady_latency_asc", "peak_memory_asc",
        "artifact_size_asc", "candidate_id_asc",
    ],
    "fast_quality_score_min": 75.0,
    "fast_max_quality_loss_points": 8.0,
    "fast_max_primary_category_loss_points": 10.0,
    "fast_min_latency_reduction_each_target": 0.30,
    "fast_min_latency_reduction_geomean": 0.35,
    "no_qualifier_result": "no-fast-selection",
}


def _no_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def load_json(path: Path | str) -> Any:
    """Load strict JSON: duplicate keys and NaN/Infinity are rejected."""

    path = Path(path)

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON number {value}")

    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_no_duplicate_pairs,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("duplicate JSON"):
            raise
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc


def _json_equal(left: Any, right: Any) -> bool:
    # Python considers True == 1, while JSON Schema does not.
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return not isinstance(left, bool) and not isinstance(right, bool) and left == right
    return type(left) is type(right) and left == right


def _type_matches(value: Any, expected: str) -> bool:
    if expected == "null":
        return value is None
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
        )
    return False


def _display_path(path: str, key: Any) -> str:
    if isinstance(key, int):
        return f"{path}[{key}]"
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", str(key)):
        return f"{path}.{key}"
    return f"{path}[{json.dumps(str(key))}]"


def _resolve_ref(ref: str, root: Mapping[str, Any]) -> Any:
    if not ref.startswith("#/"):
        raise ValueError(f"only local schema references are supported: {ref}")
    value: Any = root
    for component in ref[2:].split("/"):
        component = component.replace("~1", "/").replace("~0", "~")
        if not isinstance(value, Mapping) or component not in value:
            raise ValueError(f"unresolved schema reference: {ref}")
        value = value[component]
    return value


def _unique_json_values(values: Sequence[Any]) -> bool:
    for index, value in enumerate(values):
        if any(_json_equal(value, other) for other in values[:index]):
            return False
    return True


_RFC3339_DATE_TIME = re.compile(
    r"^(?P<year>[0-9]{4})-(?P<month>[0-9]{2})-(?P<day>[0-9]{2})"
    r"[Tt](?P<hour>[0-9]{2}):(?P<minute>[0-9]{2}):(?P<second>[0-9]{2})"
    r"(?:\.(?P<fraction>[0-9]+))?(?P<zone>[Zz]|[+-][0-9]{2}:[0-9]{2})$"
)


def _is_leap_year(year: int) -> bool:
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def _days_in_month(year: int, month: int) -> int:
    if month == 2:
        return 29 if _is_leap_year(year) else 28
    return 30 if month in {4, 6, 9, 11} else 31


def _days_before_year(year: int) -> int:
    """Days before *year* in the proleptic Gregorian calendar (year zero included)."""

    return 365 * year + (year + 3) // 4 - (year + 99) // 100 + (year + 399) // 400


def _ordinal_from_date(year: int, month: int, day: int) -> int:
    month_days = (0, 31, 29 if _is_leap_year(year) else 28, 31, 30, 31,
                  30, 31, 31, 30, 31, 30, 31)
    return _days_before_year(year) + sum(month_days[1:month]) + day - 1


def _date_from_ordinal(ordinal: int) -> tuple[int, int, int]:
    """Invert _ordinal_from_date without datetime, including negative years."""

    cycles, remainder = divmod(ordinal, 146097)  # exactly 400 Gregorian years
    year = cycles * 400
    while True:
        year_days = 366 if _is_leap_year(year) else 365
        if remainder < year_days:
            break
        remainder -= year_days
        year += 1
    month = 1
    while remainder >= _days_in_month(year, month):
        remainder -= _days_in_month(year, month)
        month += 1
    return year, month, remainder + 1


def _offset_minutes(zone: str) -> int:
    if zone in {"Z", "z"}:
        return 0
    hours = int(zone[1:3])
    minutes = int(zone[4:6])
    return (1 if zone[0] == "+" else -1) * (hours * 60 + minutes)


def _validate_rfc3339(value: str, path: str) -> None:
    """Validate structural RFC 3339, including year zero and structural leap seconds."""

    match = _RFC3339_DATE_TIME.fullmatch(value)
    if match is None:
        raise ValidationError(path, "must be an RFC 3339 date-time with Z or numeric timezone")
    parts = {key: int(match.group(key)) for key in ("year", "month", "day", "hour", "minute", "second")}
    _require(1 <= parts["month"] <= 12, path, "contains an invalid month")
    _require(1 <= parts["day"] <= _days_in_month(parts["year"], parts["month"]),
             path, "contains an invalid calendar date")
    _require(parts["hour"] <= 23, path, "contains an invalid hour")
    _require(parts["minute"] <= 59, path, "contains an invalid minute")
    _require(parts["second"] <= 60, path, "contains an invalid second")
    zone = match.group("zone")
    if zone not in {"Z", "z"}:
        offset_hour = int(zone[1:3])
        offset_minute = int(zone[4:6])
        _require(offset_hour <= 23, path, "contains an invalid timezone offset hour")
        _require(offset_minute <= 59, path, "contains an invalid timezone offset minute")

    if parts["second"] == 60:
        utc_total_minutes = (
            _ordinal_from_date(parts["year"], parts["month"], parts["day"]) * 1440
            + parts["hour"] * 60
            + parts["minute"]
            - _offset_minutes(zone)
        )
        utc_day, utc_minute = divmod(utc_total_minutes, 1440)
        utc_year, utc_month, utc_day_of_month = _date_from_ordinal(utc_day)
        _require(
            utc_month in {6, 12}
            and utc_day_of_month == _days_in_month(utc_year, utc_month)
            and utc_minute == 23 * 60 + 59,
            path,
            "leap second must map to UTC 23:59 at June or December month end",
        )


def validate(instance: Any, schema: Mapping[str, Any] | bool, *, path: str = "$", root: Mapping[str, Any] | None = None) -> None:
    """Validate *instance* against the v1 schema subset.

    This is deliberately a validator rather than a schema compiler.  Bake-off files are
    small and the direct implementation keeps failures pointing at the exact JSON path.
    """

    if schema is True:
        return
    if schema is False:
        raise ValidationError(path, "schema rejects this value")
    if not isinstance(schema, Mapping):
        raise ValueError(f"schema at {path} is not an object or boolean")
    if root is None:
        root = schema

    if "$ref" in schema:
        validate(instance, _resolve_ref(str(schema["$ref"]), root), path=path, root=root)
        # A $ref is allowed to be accompanied by annotations, but no checked-in schema
        # relies on sibling assertions.  Returning here makes that policy explicit.
        return

    if "allOf" in schema:
        for index, subschema in enumerate(schema["allOf"]):
            validate(instance, subschema, path=path, root=root)
    if "anyOf" in schema:
        errors: list[str] = []
        matched = 0
        for subschema in schema["anyOf"]:
            try:
                validate(instance, subschema, path=path, root=root)
            except ValidationError as exc:
                errors.append(str(exc))
            else:
                matched += 1
        if matched == 0:
            detail = errors[0] if errors else "no branch matched"
            raise ValidationError(path, f"anyOf failed ({detail})")
    if "oneOf" in schema:
        matched = 0
        errors: list[str] = []
        for subschema in schema["oneOf"]:
            try:
                validate(instance, subschema, path=path, root=root)
            except ValidationError as exc:
                errors.append(str(exc))
            else:
                matched += 1
        if matched != 1:
            raise ValidationError(path, f"oneOf matched {matched} branches")
    if "not" in schema:
        try:
            validate(instance, schema["not"], path=path, root=root)
        except ValidationError:
            pass
        else:
            raise ValidationError(path, "not schema matched")

    if "type" in schema:
        expected = schema["type"]
        expected_types = expected if isinstance(expected, list) else [expected]
        if not any(_type_matches(instance, str(item)) for item in expected_types):
            expected_text = ", ".join(str(item) for item in expected_types)
            raise ValidationError(path, f"expected type {expected_text}")

    if "const" in schema and not _json_equal(instance, schema["const"]):
        raise ValidationError(path, f"must equal {json.dumps(schema['const'], sort_keys=True)}")
    if "enum" in schema and not any(_json_equal(instance, item) for item in schema["enum"]):
        raise ValidationError(path, f"must be one of {json.dumps(schema['enum'])}")

    if isinstance(instance, dict):
        required = schema.get("required", [])
        for key in required:
            if key not in instance:
                raise ValidationError(path, f"missing required property {key!r}")
        properties = schema.get("properties", {})
        additional = schema.get("additionalProperties", True)
        for key, value in instance.items():
            child_path = _display_path(path, key)
            if key in properties:
                validate(value, properties[key], path=child_path, root=root)
            elif additional is False:
                raise ValidationError(path, f"unknown property {key!r}")
            elif isinstance(additional, Mapping):
                validate(value, additional, path=child_path, root=root)
        if "minProperties" in schema and len(instance) < schema["minProperties"]:
            raise ValidationError(path, f"must contain at least {schema['minProperties']} properties")
        if "maxProperties" in schema and len(instance) > schema["maxProperties"]:
            raise ValidationError(path, f"must contain at most {schema['maxProperties']} properties")

    if isinstance(instance, list):
        if "minItems" in schema and len(instance) < schema["minItems"]:
            raise ValidationError(path, f"must contain at least {schema['minItems']} items")
        if "maxItems" in schema and len(instance) > schema["maxItems"]:
            raise ValidationError(path, f"must contain at most {schema['maxItems']} items")
        if schema.get("uniqueItems") and not _unique_json_values(instance):
            raise ValidationError(path, "items must be unique")
        if "items" in schema:
            item_schema = schema["items"]
            for index, value in enumerate(instance):
                validate(value, item_schema, path=_display_path(path, index), root=root)

    if isinstance(instance, str):
        if "minLength" in schema and len(instance) < schema["minLength"]:
            raise ValidationError(path, f"must contain at least {schema['minLength']} characters")
        if "maxLength" in schema and len(instance) > schema["maxLength"]:
            raise ValidationError(path, f"must contain at most {schema['maxLength']} characters")
        if "pattern" in schema and re.search(str(schema["pattern"]), instance) is None:
            raise ValidationError(path, f"does not match pattern {schema['pattern']!r}")
        if schema.get("format") == "date-time":
            _validate_rfc3339(instance, path)

    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        value = float(instance)
        if not math.isfinite(value):
            raise ValidationError(path, "number must be finite")
        if "minimum" in schema and value < schema["minimum"]:
            raise ValidationError(path, f"must be >= {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            raise ValidationError(path, f"must be <= {schema['maximum']}")
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            raise ValidationError(path, f"must be > {schema['exclusiveMinimum']}")
        if "exclusiveMaximum" in schema and value >= schema["exclusiveMaximum"]:
            raise ValidationError(path, f"must be < {schema['exclusiveMaximum']}")
        if "multipleOf" in schema:
            quotient = value / schema["multipleOf"]
            if not math.isclose(quotient, round(quotient), rel_tol=0.0, abs_tol=1e-12):
                raise ValidationError(path, f"must be a multiple of {schema['multipleOf']}")


def _require(condition: bool, path: str, message: str) -> None:
    if not condition:
        raise ValidationError(path, message)


def _unique(items: Iterable[Any], path: str, label: str) -> None:
    values = list(items)
    _require(len(values) == len(set(values)), path, f"{label} must be unique")


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), path, "must be an object")
    return value


_FORBIDDEN_GENERATOR_KEY_WORDS = (
    "pixel", "frame", "blob", "payload", "bytes", "rgba", "rgb",
)


def _validate_metadata_only(value: Any, path: str) -> None:
    """Reject frame/pixel payloads hidden in otherwise free-form generator metadata."""

    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key).lower().replace("-", "_")
            _require(
                key_text not in {"data", "blob", "frames", "pixels"}
                and not any(word in key_text for word in _FORBIDDEN_GENERATOR_KEY_WORDS),
                f"{path}.{key}",
                "generator parameters must remain metadata-only",
            )
            _validate_metadata_only(child, f"{path}.{key}")
    elif isinstance(value, list):
        _require(False, path, "generator parameters may not contain list/blob/data payloads")
    elif isinstance(value, (bytes, bytearray)):
        _require(False, path, "generator parameters may not contain binary payloads")


def _expected_analysis_dimensions(source_width: int, source_height: int, pixel_aspect_ratio: float,
                                  cap_megapixels: float) -> tuple[int, int]:
    """Return the shared frozen ``analysisGeometry-v1`` dimensions."""

    return geometry.analysis_dimensions(
        source_width,
        source_height,
        pixel_aspect_ratio,
        cap_megapixels,
    )


def _candidate_constraint_map(protocol: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    """Return v2's candidate-specific scheduling constraints by candidate id.

    v1 intentionally has no such surface.  Callers that consume a hand-built protocol in
    unit tests also get the unconstrained legacy behaviour when the v2 field is absent; the
    checked-in v2 protocol/schema gate requires the frozen NeuFlow entry.
    """

    raw_constraints = protocol.get("candidate_constraints", [])
    if not isinstance(raw_constraints, (list, tuple)):
        return {}
    result: dict[str, Mapping[str, Any]] = {}
    for constraint in raw_constraints:
        if isinstance(constraint, Mapping) and isinstance(constraint.get("candidate_id"), str):
            result[constraint["candidate_id"]] = constraint
    return result


def _cap_map(protocol: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    caps = protocol.get("analysis_caps", [])
    if not isinstance(caps, (list, tuple)):
        return {}
    return {
        cap["token"]: cap
        for cap in caps
        if isinstance(cap, Mapping) and isinstance(cap.get("token"), str)
    }


def _candidate_constraint_geometry_ok(
    constraint: Mapping[str, Any],
    shot: Mapping[str, Any],
    cap: Mapping[str, Any],
) -> tuple[bool, str]:
    """Check a constrained candidate against one computed source/cap geometry."""

    required = constraint.get("required_geometry")
    if not isinstance(required, Mapping):
        return False, "candidate constraint has no required geometry"
    lattice = cap.get("lattice")
    if not isinstance(lattice, Mapping):
        return False, "selected cap has no frozen lattice metadata"
    for field in ("analysis_width", "analysis_height", "canonical_aspect_ratio"):
        if lattice.get(field) != required.get(field):
            return False, "candidate geometry disagrees with selected cap lattice"

    try:
        expected_width, expected_height = _expected_analysis_dimensions(
            shot["width"], shot["height"], shot["pixel_aspect_ratio"], cap["decimal_megapixels"],
        )
    except (KeyError, TypeError, ValueError, geometry.GeometryFailure):
        return False, "source geometry cannot be computed"
    if (expected_width, expected_height) != (
        required["analysis_width"], required["analysis_height"],
    ):
        return False, "computed analysis geometry is not the required fixed lattice"

    try:
        canonical_aspect = (
            float(shot["width"]) * float(shot["pixel_aspect_ratio"])
        ) / float(shot["height"])
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return False, "source aspect ratio cannot be computed"
    if not math.isfinite(canonical_aspect) or required.get("canonical_aspect_ratio") == "16:9" and not math.isclose(
        canonical_aspect, 16.0 / 9.0, rel_tol=0.0, abs_tol=1e-12,
    ):
        return False, "source geometry is not canonical 16:9"
    return True, ""


def validate_protocol_consistency(
    protocol: Mapping[str, Any],
    protocol_schema: Mapping[str, Any] | None = None,
    report_schema: Mapping[str, Any] | None = None,
) -> None:
    """Validate the executable protocol and the invariants behind its matrix."""

    if protocol_schema is not None:
        validate(protocol, protocol_schema)
    protocol_id = protocol.get("protocol_id")
    _require(protocol_id in {_V1_PROTOCOL_ID, _V2_PROTOCOL_ID}, "$.protocol_id", "wrong protocol id")
    is_v2 = protocol_id == _V2_PROTOCOL_ID
    expected_version = 2 if is_v2 else 1
    expected_date = "2026-08-23" if is_v2 else "2026-08-22"
    _require(protocol.get("schema_version") == expected_version, "$.schema_version", f"must be {expected_version}")
    _require(protocol.get("frozen_date") == expected_date, "$.frozen_date", "protocol freeze date changed")
    schema_ids = _mapping(protocol.get("schema_ids"), "$.schema_ids")
    expected_schema_ids = {
        "protocol": f"whitewater-p25-protocol-v{'2' if is_v2 else '1'}",
        "corpus": "whitewater-p25-corpus-v1",
        "report": f"whitewater-p25-report-v{'2' if is_v2 else '1'}",
    }
    _require(dict(schema_ids) == expected_schema_ids, "$.schema_ids", "schema ids do not match the protocol version")
    eligibility = protocol["eligibility"]
    _require(eligibility["required_identity"] == _EXPECTED_REQUIRED_IDENTITY,
             "$.eligibility.required_identity", "candidate identity contract changed")
    _require(eligibility["license_surfaces"] == _EXPECTED_LICENSE_SURFACES,
             "$.eligibility.license_surfaces", "licence surface contract changed")
    _require(eligibility["required_license_verdicts"] == _EXPECTED_REQUIRED_LICENSE_VERDICTS,
             "$.eligibility.required_license_verdicts", "licence verdict contract changed")
    _require(eligibility["redistribution_terms_review_required"] is True,
             "$.eligibility.redistribution_terms_review_required",
             "redistribution terms review must be required")
    _require(eligibility["required_tensor_contract"] == _EXPECTED_TENSOR_CONTRACT,
             "$.eligibility.required_tensor_contract", "tensor contract changed")

    candidates = protocol["candidate_ids"]
    _require(candidates == _EXPECTED_CANDIDATES, "$.candidate_ids", "candidate ids, roles or order changed")

    caps = protocol["analysis_caps"]
    cap_ids = [cap["token"] for cap in caps]
    _unique(cap_ids, "$.analysis_caps", "analysis cap tokens")
    expected_caps = _EXPECTED_CAPS_V2 if is_v2 else _EXPECTED_CAPS_V1
    _require(caps == expected_caps, "$.analysis_caps", "cap tokens or numeric values changed")
    _require(protocol["cap_accounting"] == _EXPECTED_CAP_ACCOUNTING,
             "$.cap_accounting", "cap accounting contract changed")

    providers = protocol["providers"]
    provider_ids = [provider["token"] for provider in providers]
    _unique(provider_ids, "$.providers", "provider tokens")
    expected_providers = _EXPECTED_PROVIDERS_V2 if is_v2 else _EXPECTED_PROVIDERS_V1
    _require(providers == expected_providers, "$.providers", "provider matrix changed")
    provider_map = {provider["token"]: provider for provider in providers}
    _require(provider_map.get("cuda", {}).get("purpose") == "selection",
             "$.providers", "CUDA must be the selection provider")
    _require(provider_map.get("cpu", {}).get("purpose") == "correctness",
             "$.providers", "CPU must be the correctness provider")
    _require(provider_map.get("coreml", {}).get("purpose") == "supporting",
             "$.providers", "CoreML must remain supporting only")

    if is_v2:
        _require(
            protocol.get("candidate_constraints") == _EXPECTED_CANDIDATE_CONSTRAINTS_V2,
            "$.candidate_constraints",
            "candidate capability constraints changed",
        )

    conditioning = protocol["conditioning"]
    conditioning_ids = [entry["token"] for entry in conditioning]
    _unique(conditioning_ids, "$.conditioning", "conditioning tokens")
    _require(conditioning == _EXPECTED_CONDITIONING, "$.conditioning",
             "conditioning tokens, accepted encodings or formulas changed")
    _require(protocol["quantile"] == _EXPECTED_QUANTILE, "$.quantile", "quantile contract changed")
    _require(protocol["padding_comparison_policy"] == _EXPECTED_PADDING_COMPARISON_POLICY,
             "$.padding_comparison_policy", "padding comparison policy changed")

    _require(protocol["synthetic_categories"] == _EXPECTED_SYNTHETIC_CATEGORIES,
             "$.synthetic_categories", "synthetic category contract changed")
    _require(protocol["primary_production_categories"] == _EXPECTED_PRODUCTION_CATEGORIES,
             "$.primary_production_categories", "production category contract changed")
    _require(protocol["metrics"] == _EXPECTED_METRICS, "$.metrics", "metric contract changed")
    if report_schema is not None:
        report_defs = _mapping(report_schema.get("$defs"), "$.report_schema.$defs")
        if is_v2:
            report_candidate = _mapping(report_defs.get("candidate"), "$.report_schema.$defs.candidate")
            report_candidate_required = report_candidate.get("required", [])
            _require(
                "measurement_status" in report_candidate_required,
                "$.report_schema.$defs.candidate.required",
                "v2 candidate measurement_status is required",
            )
            report_candidate_properties = _mapping(
                report_candidate.get("properties"), "$.report_schema.$defs.candidate.properties",
            )
            _require(
                report_candidate_properties.get("measurement_status", {}).get("enum") == _EXPECTED_MEASUREMENT_STATUSES,
                "$.report_schema.$defs.candidate.properties.measurement_status",
                "candidate measurement status values changed",
            )
            _require(
                "measurement_exclusion_reason" in report_candidate_properties,
                "$.report_schema.$defs.candidate.properties",
                "v2 candidate measurement_exclusion_reason is required",
            )
            candidate_branches = report_candidate.get("oneOf", [])
            _require(
                any(
                    "measurement_providers" in branch.get("required", [])
                    for branch in candidate_branches
                    if isinstance(branch, Mapping)
                ),
                "$.report_schema.$defs.candidate.oneOf",
                "v2 measurable candidates must declare qualified providers",
            )
            _require(
                any(
                    branch.get("not") == {"required": ["measurement_providers"]}
                    for branch in candidate_branches
                    if isinstance(branch, Mapping)
                ),
                "$.report_schema.$defs.candidate.oneOf",
                "v2 unavailable candidates must not declare provider evidence",
            )
            measurement_provider_schema = _mapping(
                report_candidate_properties.get("measurement_providers"),
                "$.report_schema.$defs.candidate.properties.measurement_providers",
            )
            _require(
                measurement_provider_schema.get("items", {}).get("enum") == ["cpu", "cuda", "coreml"],
                "$.report_schema.$defs.candidate.properties.measurement_providers",
                "candidate measurement provider values changed",
            )
        report_metrics = _mapping(report_defs.get("metrics"), "$.report_schema.$defs.metrics")
        report_metric_properties = _mapping(
            report_metrics.get("properties"), "$.report_schema.$defs.metrics.properties",
        )
        report_metric_tokens = [
            token for token in report_metric_properties if token != "not_applicable"
        ]
        _require(
            report_metric_tokens == _EXPECTED_METRICS,
            "$.report_schema.$defs.metrics.properties",
            "report metric properties diverge from frozen protocol metric tokens",
        )
        _require("not_applicable" in report_metric_properties,
                 "$.report_schema.$defs.metrics.properties.not_applicable",
                 "report metric disposition is required")
        _require("not_applicable" in report_metrics.get("required", []),
                 "$.report_schema.$defs.metrics.required",
                 "report metric disposition must be required")
    _require(protocol["review_dimensions"] == _EXPECTED_REVIEW_DIMENSIONS,
             "$.review_dimensions", "review dimensions changed")
    required_cases = protocol["required_synthetic_cases"]
    _unique(required_cases, "$.required_synthetic_cases", "required synthetic case ids")
    _require(required_cases == _EXPECTED_REQUIRED_CASES, "$.required_synthetic_cases", "synthetic case contract changed")
    _require(protocol["chain_lengths"] == [1, 2, 4, 8], "$.chain_lengths", "chain lengths changed")
    profiles = protocol["profiles"]
    _require(profiles["smoke"] == {"fresh_sessions": 1, "warmups_per_session": 0, "steady_samples_per_session": 2},
             "$.profiles.smoke", "smoke repetition contract changed")
    _require(profiles["screen"] == {"fresh_sessions": 1, "warmups_per_session": 0, "steady_samples_per_session": 5},
             "$.profiles.screen", "screen repetition contract changed")
    _require(profiles["final"] == {"fresh_sessions": 3, "warmups_per_session": 1, "steady_samples_per_session": 10},
             "$.profiles.final", "final repetition contract changed")

    _require(protocol["aggregation"] == _EXPECTED_AGGREGATION,
             "$.aggregation", "aggregation contract changed")

    _require(protocol["hard_gates"] == _EXPECTED_HARD_GATES,
             "$.hard_gates", "hard-gate values changed")
    _require(protocol["ranking"] == _EXPECTED_RANKING,
             "$.ranking", "ranking values changed")
    required_cells = protocol["hard_gates"]["mandatory_source_cap_cells"]
    seen_cells: set[tuple[str, str, str]] = set()
    for index, cell in enumerate(required_cells):
        key = (cell["source_target"], cell["cap_token"], cell["provider"])
        _require(key not in seen_cells, f"$.hard_gates.mandatory_source_cap_cells[{index}]", "duplicate mandatory cell")
        seen_cells.add(key)
        _require(cell["provider"] == "cuda", f"$.hard_gates.mandatory_source_cap_cells[{index}]", "mandatory cells must use CUDA")
        _require(cell["cap_token"] == "mp2", f"$.hard_gates.mandatory_source_cap_cells[{index}]", "mandatory cells must use mp2")
    for key, value in protocol.items():
        _require(key not in {"option_indices", "default_index", "model_option_order", "input_curve_option_order"},
                 f"$.{key}", "protocol must not publish persistent OFX option indices")


def validate_corpus_consistency(
    corpus: Mapping[str, Any],
    protocol: Mapping[str, Any],
    corpus_schema: Mapping[str, Any],
) -> None:
    """Validate corpus coverage and shot metadata against protocol tokens."""

    validate(corpus, corpus_schema)
    _require(corpus.get("schema_version") == 1, "$.schema_version", "must be 1")
    corpus_protocol_matches = corpus.get("protocol_id") == protocol["protocol_id"]
    # The v2 admission amendment does not change corpus semantics or frame identities, so it
    # deliberately reuses the already-frozen corpus-v1 document/schema.  A v1 corpus therefore
    # remains valid under the v2 protocol while every report still binds its exact corpus hash.
    if protocol.get("protocol_id") == _V2_PROTOCOL_ID:
        corpus_protocol_matches = corpus.get("protocol_id") == _V1_PROTOCOL_ID
    _require(corpus_protocol_matches, "$.protocol_id", "does not match protocol")
    partitions = corpus["partitions"]
    partition_ids = [partition["id"] for partition in partitions]
    _unique(partition_ids, "$.partitions", "partition ids")
    _require("synthetic" in partition_ids, "$.partitions", "synthetic partition is required")
    _require("production_external" in partition_ids, "$.partitions", "production_external partition is required")

    allowed_by_kind = {
        "synthetic": set(protocol["synthetic_categories"]),
        "production_external": set(protocol["primary_production_categories"]),
        "public": set(protocol["synthetic_categories"]) | set(protocol["primary_production_categories"]),
    }
    all_shot_ids: set[str] = set()
    seen_by_kind: dict[str, set[str]] = {key: set() for key in allowed_by_kind}
    for p_index, partition in enumerate(partitions):
        path = f"$.partitions[{p_index}]"
        _require(partition["id"] == partition["kind"], f"{path}.kind", "partition id and kind must match")
        kind = partition["kind"]
        if kind == "public":
            _require("terms" in partition, f"{path}.terms", "public partition must record usage terms and training overlap")
        for s_index, shot in enumerate(partition["shots"]):
            shot_path = f"{path}.shots[{s_index}]"
            for required_field in ("width", "height", "channels", "bit_depth"):
                _require(required_field in shot, f"{shot_path}.{required_field}",
                         "shot dimensions and pixel format are required")
            if "generator_parameters" in shot:
                _validate_metadata_only(shot["generator_parameters"], f"{shot_path}.generator_parameters")
            shot_id = shot["id"]
            _require(shot_id not in all_shot_ids, f"{shot_path}.id", "shot id is duplicated")
            all_shot_ids.add(shot_id)
            _require(shot["first_frame"] <= shot["reference_frame"] <= shot["last_frame"],
                     shot_path, "reference frame must be inside the frame range")
            frame_hashes = shot.get("frame_sha256", [])
            _unique(
                (frame_hash["frame"] for frame_hash in frame_hashes),
                f"{shot_path}.frame_sha256",
                "frame numbers",
            )
            for frame_index, frame_hash in enumerate(frame_hashes):
                _require(
                    shot["first_frame"] <= frame_hash["frame"] <= shot["last_frame"],
                    f"{shot_path}.frame_sha256[{frame_index}].frame",
                    "frame hash is outside the shot range",
                )
            for category in shot["categories"]:
                _require(category in allowed_by_kind[kind], f"{shot_path}.categories", f"category {category!r} is not valid for {kind}")
                seen_by_kind[kind].add(category)
            if kind == "synthetic":
                _require("case_id" in shot, f"{shot_path}.case_id", "synthetic shot must identify a frozen case")
                truth = shot.get("truth")
                _require(isinstance(truth, Mapping) and truth.get("kind") == "analytic",
                         f"{shot_path}.truth", "synthetic shots require analytic truth")
                _require(isinstance(truth.get("definition"), str) and bool(truth["definition"].strip()),
                         f"{shot_path}.truth.definition", "analytic truth needs a nonempty definition")
                case_id = shot["case_id"]
                _require(case_id in protocol["required_synthetic_cases"],
                         f"{shot_path}.case_id", f"unknown required synthetic case {shot['case_id']!r}")
                if case_id == "hdr-scene-linear":
                    _require(shot["encoding"] == "scene-linear", f"{shot_path}.encoding", "HDR scene-linear case must be scene-linear")
                elif case_id == "log-input":
                    _require(shot["encoding"] == "log", f"{shot_path}.encoding", "log-input case must be log")
                elif case_id == "par-0_5":
                    _require(math.isclose(shot["pixel_aspect_ratio"], 0.5), f"{shot_path}.pixel_aspect_ratio", "PAR 0.5 case has the wrong PAR")
                elif case_id == "par-2":
                    _require(math.isclose(shot["pixel_aspect_ratio"], 2.0), f"{shot_path}.pixel_aspect_ratio", "PAR 2 case has the wrong PAR")
                elif case_id.startswith("chain-"):
                    expected_length = int(case_id.split("-", 1)[1])
                    _require(shot.get("chain_length") == expected_length, f"{shot_path}.chain_length", "chain case length does not match case id")
                    _require(
                        shot["reference_frame"] - shot["first_frame"] >= expected_length
                        and shot["last_frame"] - shot["reference_frame"] >= expected_length,
                        shot_path,
                        "chain frame range must support the declared links on both sides of the reference",
                    )
                elif case_id == "odd-size":
                    _require(shot.get("width", 0) % 2 == 1 or shot.get("height", 0) % 2 == 1,
                             shot_path, "odd-size case must have an odd image extent")
                elif case_id == "asymmetric-padding":
                    padding = shot.get("generator_parameters", {}).get("padding", {})
                    _require(isinstance(padding, Mapping), f"{shot_path}.generator_parameters.padding", "asymmetric case needs padding metadata")
                    sides = [padding.get(side) for side in ("left", "right", "top", "bottom")]
                    _require(all(isinstance(side, int) and side >= 0 for side in sides),
                             f"{shot_path}.generator_parameters.padding", "padding sides must be nonnegative integers")
                    _require(padding["left"] != padding["right"] or padding["top"] != padding["bottom"],
                             f"{shot_path}.generator_parameters.padding", "padding must be asymmetric")
                elif case_id == "fhd-1920x1080-par1":
                    _require(
                        shot["width"] == 1920 and shot["height"] == 1080
                        and math.isclose(shot["pixel_aspect_ratio"], 1.0, rel_tol=0.0, abs_tol=1e-12),
                        shot_path, "FHD performance case must be exactly 1920x1080 PAR1",
                    )
                elif case_id == "uhd-3840x2160-par1":
                    _require(
                        shot["width"] == 3840 and shot["height"] == 2160
                        and math.isclose(shot["pixel_aspect_ratio"], 1.0, rel_tol=0.0, abs_tol=1e-12),
                        shot_path, "UHD performance case must be exactly 3840x2160 PAR1",
                    )
                _require(shot["path_pattern"].startswith("generated://"),
                         f"{shot_path}.path_pattern", "synthetic paths must identify a generator")
            if kind == "production_external":
                _require(shot["path_pattern"].lower().endswith(".exr"),
                         f"{shot_path}.path_pattern", "production corpus paths must end in .exr")

    for category in protocol["synthetic_categories"]:
        _require(category in seen_by_kind["synthetic"], "$.partitions", f"synthetic category {category!r} has no shot")
    for category in protocol["primary_production_categories"]:
        _require(category in seen_by_kind["production_external"], "$.partitions", f"production category {category!r} has no shot")
    required_cases = set(protocol["required_synthetic_cases"])
    seen_cases: list[str] = []
    for partition in partitions:
        if partition["kind"] == "synthetic":
            seen_cases.extend(shot["case_id"] for shot in partition["shots"])
    _unique(seen_cases, "$.partitions.synthetic.shots", "synthetic case ids")
    _require(set(seen_cases) == required_cases, "$.partitions.synthetic.shots",
             "synthetic case coverage must exactly match the frozen required case list")


def validate_report_consistency(
    report: Mapping[str, Any],
    protocol: Mapping[str, Any],
    report_schema: Mapping[str, Any],
    corpus: Mapping[str, Any] | None,
    corpus_schema: Mapping[str, Any],
    *,
    _corpus_already_validated: bool = False,
) -> None:
    """Validate result cells, repetition profiles and report-side hard gates.

    ``_corpus_already_validated`` is an internal driver optimization.  Ordinary callers must
    leave it false so a standalone report validation continues to validate the complete corpus
    document before resolving any report identities.
    """

    _require(corpus is not None, "$.corpus", "report validation requires the complete corpus document")
    if not _corpus_already_validated:
        # Validate the referenced corpus before resolving any report identities.  This prevents
        # a report from binding to a malformed or incomplete metadata document merely because
        # its selected shot ids happen to be present.
        validate_corpus_consistency(corpus, protocol, corpus_schema)
    validate(report, report_schema)
    is_v2 = protocol.get("protocol_id") == _V2_PROTOCOL_ID
    expected_report_version = 2 if is_v2 else 1
    _require(
        report.get("schema_version") == expected_report_version,
        "$.schema_version",
        f"must be {expected_report_version}",
    )
    _require(report.get("protocol_id") == protocol["protocol_id"], "$.protocol_id", "does not match protocol")
    runner = report.get("runner")
    _require(isinstance(runner, Mapping), "$.runner", "must be an object")
    _require(
        isinstance(runner.get("source_commit"), str)
        and re.fullmatch(r"[0-9a-f]{40}", runner["source_commit"]) is not None,
        "$.runner.source_commit",
        "runner source_commit must be lowercase 40-hex",
    )
    _require(report.get("corpus_id") == corpus["corpus_id"], "$.corpus_id", "does not match corpus")
    _require(
        report.get("corpus_sha256") == canonical_sha256(corpus),
        "$.corpus_sha256",
        "does not match the canonical SHA256 of the complete corpus",
    )
    candidate_entries = report["candidates"]
    candidate_ids = [entry["candidate_id"] for entry in candidate_entries]
    _unique(candidate_ids, "$.candidates", "candidate ids")
    candidate_map = {entry["candidate_id"]: entry for entry in candidate_entries}
    known_candidates = {entry["id"]: entry for entry in protocol["candidate_ids"]}
    protocol_provider_tokens = {entry["token"] for entry in protocol["providers"]}
    for candidate_id, candidate in candidate_map.items():
        _require(candidate_id in known_candidates, "$.candidates", f"unknown candidate {candidate_id!r}")
        if is_v2:
            role = known_candidates[candidate_id]["role"]
            measurement_status = candidate["measurement_status"]
            _require(
                measurement_status in set(_EXPECTED_MEASUREMENT_STATUSES),
                f"$.candidates[{candidate_id}].measurement_status",
                "measurement_status must be measurable or unavailable",
            )
            if measurement_status == "unavailable":
                _require(
                    "measurement_exclusion_reason" in candidate,
                    f"$.candidates[{candidate_id}]",
                    "unavailable candidate needs a typed measurement exclusion reason",
                )
                _require(
                    "measurement_providers" not in candidate,
                    f"$.candidates[{candidate_id}].measurement_providers",
                    "unavailable candidate must not carry provider measurement evidence",
                )
            else:
                _require(
                    "measurement_exclusion_reason" not in candidate,
                    f"$.candidates[{candidate_id}].measurement_exclusion_reason",
                    "measurable candidate must not carry a measurement exclusion reason",
                )
            if candidate["status"] == "eligible":
                _require(
                    role == "shipping-candidate",
                    f"$.candidates[{candidate_id}].status",
                    "validation-baseline candidates cannot be shipping-eligible",
                )
                _require(
                    measurement_status == "measurable",
                    f"$.candidates[{candidate_id}].measurement_status",
                    "shipping-eligible candidates must be measurable",
                )
            if measurement_status == "measurable":
                for field in (*_EXPECTED_REQUIRED_IDENTITY, "manifest_sha256", "artifact_size_bytes"):
                    _require(
                        field in candidate,
                        f"$.candidates[{candidate_id}]",
                        f"measurable candidate needs {field}",
                    )
                measurement_providers = candidate.get("measurement_providers")
                _require(
                    isinstance(measurement_providers, list)
                    and bool(measurement_providers)
                    and len(measurement_providers) == len(set(measurement_providers))
                    and all(provider in protocol_provider_tokens for provider in measurement_providers),
                    f"$.candidates[{candidate_id}].measurement_providers",
                    "measurable candidate must list unique known qualified providers",
                )
                if candidate["status"] == "excluded":
                    for field in (
                        "license_verdicts",
                        "redistribution_permitted",
                        "redistribution_terms_reviewed",
                    ):
                        _require(
                            field in candidate,
                            f"$.candidates[{candidate_id}]",
                            f"excluded measurable candidate needs {field} for legal comparison evidence",
                        )
        if candidate["status"] == "eligible":
            _require(
                "exclusion_reason" not in candidate,
                f"$.candidates[{candidate_id}].exclusion_reason",
                "shipping-eligible candidate must not carry a shipping exclusion reason",
            )
            _require(
                isinstance(candidate.get("source_commit"), str)
                and re.fullmatch(r"[0-9a-f]{40}", candidate["source_commit"]) is not None,
                f"$.candidates[{candidate_id}].source_commit",
                "eligible candidate source_commit must be lowercase 40-hex",
            )
            _require("artifact_size_bytes" in candidate,
                     f"$.candidates[{candidate_id}]", "eligible candidate needs artifact size")
            verdicts = candidate["license_verdicts"]
            _require(all(value == "commercial_use_permitted" for value in verdicts.values()),
                     f"$.candidates[{candidate_id}].license_verdicts", "eligible candidate has a non-permitted licence surface")
            redistribution = candidate["redistribution_permitted"]
            _require(all(value == "permitted" for value in redistribution.values()),
                     f"$.candidates[{candidate_id}].redistribution_permitted",
                     "eligible candidate has a non-permitted redistribution surface")
            _require(all(candidate["redistribution_terms_reviewed"].values()),
                     f"$.candidates[{candidate_id}].redistribution_terms_reviewed",
                     "eligible candidate needs reviewed redistribution terms for every surface")
        if candidate["status"] == "excluded":
            _require("exclusion_reason" in candidate, f"$.candidates[{candidate_id}]", "excluded candidate needs a typed reason")

    corpus_shots: dict[str, Mapping[str, Any]] = {}
    corpus_synthetic_shot_ids: set[str] = set()
    for partition in corpus["partitions"]:
        for shot in partition["shots"]:
            corpus_shots[shot["id"]] = shot
            if partition["kind"] == "synthetic":
                corpus_synthetic_shot_ids.add(shot["id"])

    provider_map = {provider["token"]: provider for provider in protocol["providers"]}
    cap_ids = {cap["token"] for cap in protocol["analysis_caps"]}
    cap_map = {cap["token"]: cap for cap in protocol["analysis_caps"]}
    conditioning_map = {entry["token"]: entry for entry in protocol["conditioning"]}
    conditioning_ids = {entry["token"] for entry in protocol["conditioning"]}
    profile = report["profile"]
    expected_profile = protocol["profiles"][profile]

    # A report may deliberately screen a smaller subset than the full protocol, but the
    # subset itself is part of the signed report contract.  Hashing the selector catches
    # accidental edits while the Cartesian identity check below catches dropped rows.
    matrix = _mapping(report["matrix"], "$.matrix")
    matrix_payload = {key: value for key, value in matrix.items() if key != "matrix_sha256"}
    _require(
        canonical_sha256(matrix_payload) == matrix["matrix_sha256"],
        "$.matrix.matrix_sha256",
        "does not match the canonical matrix selector",
    )
    matrix_candidate_ids = list(matrix["candidate_ids"])
    matrix_shot_ids = list(matrix["shot_ids"])
    matrix_conditioning_tokens = list(matrix["conditioning_tokens"])
    matrix_cap_tokens = list(matrix["cap_tokens"])
    _unique(matrix_candidate_ids, "$.matrix.candidate_ids", "candidate ids")
    _unique(matrix_shot_ids, "$.matrix.shot_ids", "shot ids")
    _unique(matrix_conditioning_tokens, "$.matrix.conditioning_tokens", "conditioning tokens")
    _unique(matrix_cap_tokens, "$.matrix.cap_tokens", "cap tokens")
    for candidate_index, candidate_id in enumerate(matrix_candidate_ids):
        _require(candidate_id in candidate_map, f"$.matrix.candidate_ids[{candidate_index}]",
                 "candidate is not declared in the report")
        if is_v2:
            _require(
                candidate_map[candidate_id]["measurement_status"] == "measurable",
                f"$.matrix.candidate_ids[{candidate_index}]",
                "candidates without a measurable technical artifact cannot be selected for measurement",
            )
        else:
            _require(candidate_map[candidate_id]["status"] == "eligible",
                     f"$.matrix.candidate_ids[{candidate_index}]",
                     "excluded candidates cannot be selected for measurement")
    for shot_index, shot_id in enumerate(matrix_shot_ids):
        if corpus_shots:
            _require(shot_id in corpus_shots, f"$.matrix.shot_ids[{shot_index}]",
                     "shot is absent from corpus")
    for conditioning_index, conditioning_token in enumerate(matrix_conditioning_tokens):
        _require(conditioning_token in conditioning_ids,
                 f"$.matrix.conditioning_tokens[{conditioning_index}]",
                 "unknown conditioning token")
    for cap_index, cap_token in enumerate(matrix_cap_tokens):
        _require(cap_token in cap_ids, f"$.matrix.cap_tokens[{cap_index}]", "unknown cap token")

    matrix_provider_entries = list(matrix["providers"])
    matrix_provider_tokens = [entry["token"] for entry in matrix_provider_entries]
    _unique(matrix_provider_tokens, "$.matrix.providers", "provider tokens")
    matrix_provider_loads: list[tuple[str, str]] = []
    for provider_index, entry in enumerate(matrix_provider_entries):
        provider_path = f"$.matrix.providers[{provider_index}]"
        provider_token = entry["token"]
        provider = provider_map.get(provider_token)
        _require(provider is not None, f"{provider_path}.token", "unknown provider")
        host_loads = list(entry["host_loads"])
        _unique(host_loads, f"{provider_path}.host_loads", "host loads")
        if provider_token != "cuda":
            _require(host_loads == ["not_applicable"], f"{provider_path}.host_loads",
                     "CPU/support providers must select not_applicable")
        elif profile == "final":
            _require(set(host_loads) == {"idle", "live_flame"}, f"{provider_path}.host_loads",
                     "final CUDA selectors must include idle and live_flame")
        for cap_index, cap_token in enumerate(matrix_cap_tokens):
            _require(cap_token in provider["cap_tokens"],
                     f"$.matrix.cap_tokens[{cap_index}]",
                     f"provider does not support selected cap {cap_token!r}")
        matrix_provider_loads.extend((provider_token, host_load) for host_load in host_loads)

    # Candidate constraints are checked before expanding the Cartesian matrix.  In particular,
    # a fixed-shape NeuFlow row must never be emitted and then marked as a runtime failure for a
    # cap, provider, or source geometry that the graph cannot represent.
    candidate_constraints = _candidate_constraint_map(protocol) if is_v2 else {}
    for candidate_index, candidate_id in enumerate(matrix_candidate_ids):
        candidate = candidate_map[candidate_id]
        measurement_providers = candidate.get("measurement_providers")
        if is_v2 and isinstance(measurement_providers, list):
            for provider_token in matrix_provider_tokens:
                _require(
                    provider_token in measurement_providers,
                    f"$.matrix.candidate_ids[{candidate_index}]",
                    f"candidate {candidate_id!r} has no technical measurement evidence for provider {provider_token!r}",
                )
        constraint = candidate_constraints.get(candidate_id)
        if constraint is None:
            continue
        allowed_providers = constraint.get("providers", [])
        allowed_caps = constraint.get("cap_tokens", [])
        for provider_token in matrix_provider_tokens:
            _require(
                provider_token in allowed_providers,
                f"$.matrix.providers",
                f"candidate {candidate_id!r} does not support provider {provider_token!r}",
            )
        for cap_index, cap_token in enumerate(matrix_cap_tokens):
            _require(
                cap_token in allowed_caps,
                f"$.matrix.cap_tokens[{cap_index}]",
                f"candidate {candidate_id!r} does not support cap {cap_token!r}",
            )
            cap = cap_map[cap_token]
            for shot_index, shot_id in enumerate(matrix_shot_ids):
                shot = corpus_shots[shot_id]
                valid, reason = _candidate_constraint_geometry_ok(constraint, shot, cap)
                _require(
                    valid,
                    f"$.matrix.shot_ids[{shot_index}]",
                    f"candidate {candidate_id!r} cannot use shot {shot_id!r} at cap {cap_token!r}: {reason}",
                )

    hardware = report["hardware"]
    selected_provider_tokens = set(matrix_provider_tokens)
    selected_environments = {provider_map[token]["environment"] for token in selected_provider_tokens}
    _require(len(selected_environments) == 1, "$.matrix.providers",
             "all selected providers must share one frozen environment")
    selected_environment = next(iter(selected_environments))
    _require(report["environment"] == selected_environment, "$.environment",
             "report environment does not match selected provider environment")
    platform = str(hardware.get("platform", "")).lower()
    architecture = str(hardware.get("architecture", "")).lower()
    if selected_environment == "el8-x86_64":
        _require("linux" in platform and "darwin" not in platform and "macos" not in platform,
                 "$.hardware.platform", "EL8 reports require a Linux platform")
        _require(architecture in {"x86_64", "amd64"}, "$.hardware.architecture",
                 "EL8 reports require x86_64 architecture")
    else:
        _require("macos" in platform or "darwin" in platform, "$.hardware.platform",
                 "macOS reports require a macOS platform")
        _require(architecture in {"arm64", "aarch64"}, "$.hardware.architecture",
                 "macOS reports require arm64 architecture")
    if "cuda" in selected_provider_tokens:
        for field in ("gpu", "driver", "os_release"):
            _require(field in hardware and hardware[field], f"$.hardware.{field}",
                     "CUDA matrix requires GPU, driver and OS identity")
    if "cpu" in selected_provider_tokens:
        for field in ("cpu", "os_release"):
            _require(field in hardware and hardware[field], f"$.hardware.{field}",
                     "CPU matrix requires CPU and OS identity")
    if "coreml" in selected_provider_tokens:
        _require("macos" in platform or "darwin" in platform, "$.hardware.platform",
                 "CoreML matrix requires a macOS platform")
        for field in ("cpu", "os_release"):
            _require(field in hardware and hardware[field], f"$.hardware.{field}",
                     "CoreML matrix requires CPU and OS identity")

    if profile == "final":
        _require(corpus_shots, "$.matrix", "final coverage requires corpus shot metadata")
        _require("cuda" in selected_provider_tokens, "$.matrix.providers",
                 "final reports must include CUDA coverage")
        _require("mp2" in matrix_cap_tokens, "$.matrix.cap_tokens",
                 "final reports must include the mp2 performance cap")
        for target_case, target_width, target_height, target_label in (
            ("fhd-1920x1080-par1", 1920, 1080, "FHD"),
            ("uhd-3840x2160-par1", 3840, 2160, "UHD"),
        ):
            _require(
                any(
                    shot_id in matrix_shot_ids
                    and shot_id in corpus_synthetic_shot_ids
                    and corpus_shots[shot_id].get("case_id") == target_case
                    and corpus_shots[shot_id].get("width") == target_width
                    and corpus_shots[shot_id].get("height") == target_height
                    and math.isclose(corpus_shots[shot_id].get("pixel_aspect_ratio", 0.0), 1.0, rel_tol=0.0, abs_tol=1e-12)
                    for shot_id in corpus_shots
                ),
                "$.matrix.shot_ids",
                f"final matrix needs an exact {target_label} PAR1 performance shot",
            )

    expected_cells = {
        (candidate_id, shot_id, conditioning_token, cap_token, provider_token, host_load)
        for candidate_id, shot_id, conditioning_token, cap_token, (provider_token, host_load)
        in product(
            matrix_candidate_ids,
            matrix_shot_ids,
            matrix_conditioning_tokens,
            matrix_cap_tokens,
            matrix_provider_loads,
        )
    }
    seen_cells: set[tuple[str, str, str, str, str, str]] = set()
    final_cuda_cells: dict[tuple[str, str, str, str, str], set[str]] = {}
    status_counts = {"pass": 0, "fail": 0, "skip": 0}
    for index, result in enumerate(report["results"]):
        path = f"$.results[{index}]"
        candidate_id = result["candidate_id"]
        _require(candidate_id in candidate_map, f"{path}.candidate_id", "result candidate is not declared")
        _require(result["conditioning_token"] in conditioning_ids, f"{path}.conditioning_token", "unknown conditioning token")
        _require(result["cap_token"] in cap_ids, f"{path}.cap_token", "unknown cap token")
        provider = provider_map.get(result["provider"])
        _require(provider is not None, f"{path}.provider", "unknown provider")
        _require(result["cap_token"] in provider["cap_tokens"], f"{path}", "provider does not support cap token")
        shot: Mapping[str, Any] | None = None
        if corpus_shots:
            _require(result["shot_id"] in corpus_shots, f"{path}.shot_id", "result shot is absent from corpus")
            shot = corpus_shots[result["shot_id"]]
            if "category" in result:
                _require(result["category"] in shot["categories"], f"{path}.category", "category is not declared by shot")
        key = (
            candidate_id,
            result["shot_id"],
            result["conditioning_token"],
            result["cap_token"],
            result["provider"],
            result["host_load"],
        )
        _require(key not in seen_cells, path, "duplicate matrix cell")
        seen_cells.add(key)
        status = result["status"]
        status_counts[status] += 1
        if result["provider"] != "cuda":
            _require(result["host_load"] == "not_applicable", f"{path}.host_load",
                     "CPU/support cells must use not_applicable host load")
        if profile == "final" and result["provider"] == "cuda":
            _require(result["host_load"] in {"idle", "live_flame"}, f"{path}.host_load",
                     "final CUDA cells must identify idle or live_flame load")
            pair_key = (
                candidate_id,
                result["shot_id"],
                result["conditioning_token"],
                result["cap_token"],
                result["provider"],
            )
            final_cuda_cells.setdefault(pair_key, set()).add(result["host_load"])
        if "metrics" in result:
            _validate_metric_disposition(result["metrics"], protocol, path)
        if status == "pass":
            _require("failure" not in result, path, "passing result must not carry a failure")
            for field in ("input_frames", "geometry", "timing", "metrics", "resource", "environment"):
                _require(field in result, path, f"passing result needs {field}")
            _require(shot is not None, f"{path}.shot_id", "pass geometry requires corpus shot metadata")
            _validate_conditioning(result, shot, conditioning_map, path)
            _validate_result_measurement(
                result, shot, path, expected_profile, protocol, cap_map, profile,
            )
        else:
            _require("failure" in result, path, "non-pass result needs a typed failure")

    summary = report["summary"]
    _require(summary["required_cells"] == len(report["results"]),
             "$.summary.required_cells", "must equal the number of result rows")
    _require(summary["passed_cells"] == status_counts["pass"], "$.summary.passed_cells", "does not match results")
    _require(summary["failed_cells"] == status_counts["fail"], "$.summary.failed_cells", "does not match results")
    _require(summary["skipped_cells"] == status_counts["skip"], "$.summary.skipped_cells", "does not match results")

    for pair_key, loads in final_cuda_cells.items():
        _require(loads == {"idle", "live_flame"},
                 "$.results", f"final CUDA cell {pair_key!r} needs paired idle/live_flame rows")

    missing_cells = expected_cells - seen_cells
    extra_cells = seen_cells - expected_cells
    _require(not missing_cells, "$.results",
             f"missing matrix result identities: {sorted(missing_cells)!r}")
    _require(not extra_cells, "$.results",
             f"result identities are outside the declared matrix: {sorted(extra_cells)!r}")

def _validate_conditioning(
    result: Mapping[str, Any],
    shot: Mapping[str, Any],
    conditioning_map: Mapping[str, Mapping[str, Any]],
    path: str,
) -> None:
    token = result["conditioning_token"]
    conditioning = conditioning_map[token]
    encoding = shot["encoding"]
    accepted_encoding = conditioning["accepted_encoding"]
    _require(
        accepted_encoding == "scene-linear-or-log" or accepted_encoding == encoding,
        f"{path}.conditioning_token",
        f"conditioning token is incompatible with {encoding} shot encoding",
    )
    parameters = result.get("conditioning_parameters")
    if token == "pair-percentile-v1":
        _require(isinstance(parameters, Mapping), f"{path}.conditioning_parameters",
                 "pair-percentile passes require low/high/epsilon parameters")
        _require(set(parameters) == {"low", "high", "epsilon"},
                 f"{path}.conditioning_parameters", "pair-percentile parameters are incomplete")
        low = parameters["low"]
        high = parameters["high"]
        epsilon = parameters["epsilon"]
        _require(isinstance(low, (int, float)) and not isinstance(low, bool) and math.isfinite(float(low)),
                 f"{path}.conditioning_parameters.low", "low must be finite")
        _require(isinstance(high, (int, float)) and not isinstance(high, bool) and math.isfinite(float(high)),
                 f"{path}.conditioning_parameters.high", "high must be finite")
        _require(float(high) > float(low), f"{path}.conditioning_parameters", "high must exceed low")
        _require(epsilon == 1e-6, f"{path}.conditioning_parameters.epsilon", "epsilon must be 1e-6")
    else:
        _require(parameters is None or parameters == {}, f"{path}.conditioning_parameters",
                 "non-percentile curves do not accept conditioning parameters")


def _validate_result_measurement(
    result: Mapping[str, Any],
    shot: Mapping[str, Any],
    path: str,
    expected_profile: Mapping[str, int],
    protocol: Mapping[str, Any],
    cap_map: Mapping[str, Mapping[str, Any]],
    profile: str,
) -> None:
    geometry = result["geometry"]
    _require(geometry["source_width"] == shot["width"], f"{path}.geometry.source_width",
             "source width does not match corpus shot")
    _require(geometry["source_height"] == shot["height"], f"{path}.geometry.source_height",
             "source height does not match corpus shot")
    _require(math.isclose(geometry["source_pixel_aspect_ratio"], shot["pixel_aspect_ratio"],
                          rel_tol=0.0, abs_tol=1e-12),
             f"{path}.geometry.source_pixel_aspect_ratio", "source PAR does not match corpus shot")
    expected_canonical_width = shot["width"] * shot["pixel_aspect_ratio"]
    expected_canonical_height = shot["height"]
    _require(math.isclose(geometry["canonical_width"], expected_canonical_width,
                          rel_tol=0.0, abs_tol=1e-9),
             f"{path}.geometry.canonical_width", "canonical width does not match source extent and PAR")
    _require(math.isclose(geometry["canonical_height"], expected_canonical_height,
                          rel_tol=0.0, abs_tol=1e-9),
             f"{path}.geometry.canonical_height", "canonical height does not match source extent")
    expected_analysis_width, expected_analysis_height = _expected_analysis_dimensions(
        shot["width"], shot["height"], shot["pixel_aspect_ratio"],
        cap_map[result["cap_token"]]["decimal_megapixels"],
    )
    _require(geometry["analysis_width"] == expected_analysis_width,
             f"{path}.geometry.analysis_width", "analysis width does not match cap sizing contract")
    _require(geometry["analysis_height"] == expected_analysis_height,
             f"{path}.geometry.analysis_height", "analysis height does not match cap sizing contract")
    _require(geometry["padded_width"] >= geometry["analysis_width"], f"{path}.geometry", "padded width is below analysis width")
    _require(geometry["padded_height"] >= geometry["analysis_height"], f"{path}.geometry", "padded height is below analysis height")
    _require(math.isclose(
        geometry["spacing_x_source_pixels"],
        shot["width"] / geometry["analysis_width"],
        rel_tol=0.0,
        abs_tol=1e-12,
    ), f"{path}.geometry.spacing_x_source_pixels", "does not bind source width to analysis width")
    _require(math.isclose(
        geometry["spacing_y_source_pixels"],
        shot["height"] / geometry["analysis_height"],
        rel_tol=0.0,
        abs_tol=1e-12,
    ), f"{path}.geometry.spacing_y_source_pixels", "does not bind source height to analysis height")
    expected_megapixels = geometry["padded_width"] * geometry["padded_height"] / protocol["cap_accounting"]["unit_pixels"]
    _require(math.isclose(geometry["effective_padded_megapixels"], expected_megapixels, rel_tol=0.0, abs_tol=1e-9),
             f"{path}.geometry.effective_padded_megapixels", "does not equal padded tensor area")
    timing = result["timing"]
    sessions = timing["sessions"]
    _require(len(sessions) == expected_profile["fresh_sessions"], f"{path}.timing.sessions", "fresh-session count does not match profile")
    for session_index, session in enumerate(sessions):
        session_path = f"{path}.timing.sessions[{session_index}]"
        _require(session["session_index"] == session_index, f"{session_path}.session_index", "session indices must be contiguous")
        needs_warmup = expected_profile["warmups_per_session"] > 0
        _require(session["warmup_recorded"] == needs_warmup, f"{session_path}.warmup_recorded", "warm-up recording does not match profile")
        _require(len(session["steady_samples_ms"]) == expected_profile["steady_samples_per_session"],
                 f"{session_path}.steady_samples_ms", "steady repetition count does not match profile")
        if needs_warmup:
            _require("warmup_ms" in session, f"{session_path}", "recorded warm-up needs a duration")
    concatenated_steady = [
        sample
        for session in sessions
        for sample in session["steady_samples_ms"]
    ]
    _require(timing["steady_samples_ms"] == concatenated_steady,
             f"{path}.timing.steady_samples_ms",
             "top-level steady samples must equal the per-session concatenation")
    _require(math.isclose(
        timing["session_creation_ms"],
        metrics.linear_quantile(
            [session["session_creation_ms"] for session in sessions], 0.5
        ),
        rel_tol=0.0,
        abs_tol=1e-9,
    ),
             f"{path}.timing.session_creation_ms",
             "must be the median of per-session creation durations")
    _require(math.isclose(
        timing["first_inference_ms"],
        metrics.linear_quantile(
            [session["first_inference_ms"] for session in sessions], 0.5
        ),
        rel_tol=0.0,
        abs_tol=1e-9,
    ),
             f"{path}.timing.first_inference_ms",
             "must be the median of per-session first-inference durations")
    _require(math.isclose(
        timing["steady_inference_ms"],
        metrics.linear_quantile(concatenated_steady, 0.5),
        rel_tol=0.0,
        abs_tol=1e-9,
    ),
             f"{path}.timing.steady_inference_ms",
             "must be the median of flattened steady samples")
    expected_total_pair_ms = (
        timing["preprocessing_ms"]
        + timing["steady_inference_ms"]
        + timing["postprocessing_ms"]
    )
    _require(math.isclose(timing["total_pair_ms"], expected_total_pair_ms,
                          rel_tol=0.0, abs_tol=1e-9),
             f"{path}.timing.total_pair_ms",
             "must equal preprocessing plus steady inference plus postprocessing")
    report_metrics = result["metrics"]
    not_applicable = report_metrics["not_applicable"]
    _validate_metric_applicability(report_metrics, not_applicable, shot, path)

    values_by_section = {"metrics": report_metrics, "resource": result["resource"]}
    for gate in PER_CELL_HARD_GATES:
        values = values_by_section[gate.result_section]
        _require(
            values[gate.result_key] <= protocol["hard_gates"][gate.protocol_key],
            f"{path}.{gate.result_section}.{gate.result_key}",
            gate.validation_message,
        )
    input_frames = result["input_frames"]
    _require(len(input_frames) == 2, f"{path}.input_frames", "exactly two input frames are required")
    frame_numbers = [frame["frame"] for frame in input_frames]
    _require(len(set(frame_numbers)) == 2, f"{path}.input_frames", "input frames must be distinct")
    _require(all(shot["first_frame"] <= frame <= shot["last_frame"] for frame in frame_numbers),
             f"{path}.input_frames", "input frames must be inside the corpus shot range")
    corpus_frame_hashes = {entry["frame"]: entry["sha256"] for entry in shot.get("frame_sha256", [])}
    for frame_index, frame in enumerate(input_frames):
        _require("sha256" in frame, f"{path}.input_frames[{frame_index}]", "input frame identity is required")
        if frame["frame"] in corpus_frame_hashes:
            _require(frame["sha256"] == corpus_frame_hashes[frame["frame"]],
                     f"{path}.input_frames[{frame_index}].sha256",
                     "does not match the corpus frame hash")
    if profile == "final" and result["provider"] == "cuda":
        stages = {sample["stage"] for sample in result["resource"].get("nvml_samples", [])}
        required_stages = {"baseline", "session_create", "steady", "cleanup", "process_exit"}
        _require(result["host_load"] in {"idle", "live_flame"},
                 f"{path}.host_load", "final CUDA pass must identify idle or live_flame load")
        _require(required_stages <= stages, f"{path}.resource.nvml_samples",
                 "final CUDA pass needs baseline/session_create/steady/cleanup/process_exit NVML stages")


def _validate_metric_disposition(
    metrics: Mapping[str, Any],
    protocol: Mapping[str, Any],
    path: str,
) -> None:
    """Require every frozen metric to be numeric or explicitly not applicable."""

    metric_tokens = set(protocol["metrics"])
    not_applicable = metrics["not_applicable"]
    _unique(not_applicable, f"{path}.metrics.not_applicable", "metric dispositions")
    unknown_dispositions = set(not_applicable) - metric_tokens
    _require(not unknown_dispositions, f"{path}.metrics.not_applicable",
             f"unknown metric dispositions: {sorted(unknown_dispositions)!r}")
    present_metrics = {
        metric_name for metric_name in metrics
        if metric_name != "not_applicable" and metric_name in metric_tokens
    }
    overlap = present_metrics & set(not_applicable)
    _require(not overlap, f"{path}.metrics.not_applicable",
             f"metrics cannot be both numeric and not_applicable: {sorted(overlap)!r}")
    _require(present_metrics | set(not_applicable) == metric_tokens,
             f"{path}.metrics", "every protocol metric must be numeric or explicitly not_applicable")
    for metric_name in present_metrics:
        value = metrics[metric_name]
        _require(
            isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)),
            f"{path}.metrics.{metric_name}", "numeric metric must be finite",
        )


def _validate_metric_applicability(
    metrics: Mapping[str, Any],
    not_applicable: Sequence[str],
    shot: Mapping[str, Any],
    path: str,
) -> None:
    """Apply truth/reliability applicability rules to a measured pass row."""

    def require_numeric_metric(metric_name: str, reason: str) -> None:
        _require(metric_name in metrics and metric_name not in not_applicable,
                 f"{path}.metrics.{metric_name}", reason)

    require_numeric_metric("nonfinite_fraction", "this reliability metric is always applicable")
    require_numeric_metric("repeated_run_p99_delta_px", "this reliability metric is always applicable")
    truth = shot.get("truth")
    truth_kind = truth.get("kind") if isinstance(truth, Mapping) else None
    if truth_kind == "analytic":
        for metric_name in ("endpoint_error_px", "fraction_le_1px", "fraction_le_3px"):
            require_numeric_metric(metric_name, "analytic dense-truth pass requires this metric")
    if truth_kind == "landmarks":
        for metric_name in ("landmark_median_error_px", "landmark_p95_error_px"):
            require_numeric_metric(metric_name, "landmark truth requires this metric")
    if "chain_length" in shot:
        require_numeric_metric("chain_drift_px", "chain shots require chain drift")


def validate_protocol_and_schemas(
    protocol: Mapping[str, Any],
    protocol_schema: Mapping[str, Any],
    corpus_schema: Mapping[str, Any],
    report_schema: Mapping[str, Any],
) -> None:
    """Check schema IDs and the protocol's references before validating fixtures."""

    validate_protocol_consistency(protocol, protocol_schema, report_schema)
    is_v2 = protocol.get("protocol_id") == _V2_PROTOCOL_ID
    version = 2 if is_v2 else 1
    suffix = "2" if is_v2 else "1"
    expected_ids = {
        "protocol": f"whitewater://schema/phase2.5/protocol-v{suffix}",
        "corpus": "whitewater://schema/phase2.5/corpus-v1",
        "report": f"whitewater://schema/phase2.5/report-v{suffix}",
    }
    schemas = {"protocol": protocol_schema, "corpus": corpus_schema, "report": report_schema}
    for name, schema in schemas.items():
        _require(schema.get("$id") == expected_ids[name], f"$.schemas.{name}.$id", "schema id does not match the selected protocol version")
        _require(schema.get("$schema") == "https://json-schema.org/draft/2020-12/schema",
                 f"$.schemas.{name}.$schema", "schema dialect is not pinned")
    expected_corpus_protocol = _V1_PROTOCOL_ID if is_v2 else protocol["protocol_id"]
    _require(corpus_schema["properties"]["protocol_id"]["const"] == expected_corpus_protocol,
             "$.schemas.corpus.protocol_id", "corpus schema protocol id diverges")
    _require(report_schema["properties"]["protocol_id"]["const"] == protocol["protocol_id"],
             "$.schemas.report.protocol_id", "report schema protocol id diverges")
    _require(corpus_schema["properties"]["schema_version"]["const"] == 1,
             "$.schemas.corpus.schema_version", "corpus schema version diverges")
    _require(report_schema["properties"]["schema_version"]["const"] == version,
             "$.schemas.report.schema_version", "report schema version diverges")
