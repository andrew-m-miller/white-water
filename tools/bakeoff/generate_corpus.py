#!/usr/bin/env python3
"""Build a complete Phase 2.5 corpus document and optional synthetic frames.

The output corpus contains the frozen synthetic partition plus the production shot
metadata from ``bakeoff/production-corpus-v1.template.json``.  The template is a
deliberate partition fragment: operators replace its paths/metadata on the air-gapped
box and then use this script to produce the validator-bound complete corpus.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from synthetic import REQUIRED_SYNTHETIC_CASES, synthetic_partition, write_case_frames  # type: ignore
else:
    from .synthetic import REQUIRED_SYNTHETIC_CASES, synthetic_partition, write_case_frames


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_production_partition(path: Path) -> Mapping[str, Any]:
    template = load_json(path)
    if not isinstance(template, Mapping):
        raise ValueError("production template must be a JSON object")
    partition = template.get("partition")
    if not isinstance(partition, Mapping):
        raise ValueError("production template must contain a partition object")
    if partition.get("id") != "production_external" or partition.get("kind") != "production_external":
        raise ValueError("production template partition must be production_external")
    return partition


def build_corpus(production_template: Path, *, corpus_id: str) -> dict[str, Any]:
    partition = dict(load_production_partition(production_template))
    corpus = {
        "schema_version": 1,
        "protocol_id": "whitewater-p25-v1",
        "corpus_id": corpus_id,
        "description": (
            "Phase 2.5 corpus: deterministic synthetic analytic-truth cases plus "
            "anonymous production metadata."
        ),
        "partitions": [synthetic_partition(), partition],
    }
    synthetic_case_ids = [shot["case_id"] for shot in corpus["partitions"][0]["shots"]]
    if tuple(synthetic_case_ids) != REQUIRED_SYNTHETIC_CASES:
        raise AssertionError("generated synthetic corpus does not cover the frozen case list")
    return corpus


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[2]
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="complete corpus JSON output path",
    )
    parser.add_argument(
        "--production-template",
        type=Path,
        default=root / "bakeoff/production-corpus-v1.template.json",
        help="external production partition template",
    )
    parser.add_argument(
        "--corpus-id",
        default="p25-generated-corpus",
        help="stable corpus id for this generated document",
    )
    parser.add_argument(
        "--frames-dir",
        type=Path,
        help="optional directory for generated PFM frames and truth sidecars",
    )
    parser.add_argument(
        "--include-large",
        action="store_true",
        help="also emit the 1920x1080 and 3840x2160 cases (large output)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    corpus = build_corpus(args.production_template, corpus_id=args.corpus_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(corpus, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.frames_dir is not None:
        for shot in corpus["partitions"][0]["shots"]:
            write_case_frames(shot["case_id"], args.frames_dir, include_large=args.include_large)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
