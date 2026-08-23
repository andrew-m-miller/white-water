#!/usr/bin/env python3
"""Command-line entry point for the Phase 2.5 schema and protocol gate."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from validator import (  # type: ignore
        ValidationError,
        load_json,
        validate,
        validate_corpus_consistency,
        validate_protocol_and_schemas,
        validate_protocol_consistency,
        validate_report_consistency,
    )
else:
    from .validator import (
        ValidationError,
        load_json,
        validate,
        validate_corpus_consistency,
        validate_protocol_and_schemas,
        validate_protocol_consistency,
        validate_report_consistency,
    )


def _default_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    root = _default_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("document", type=Path, help="protocol, corpus or report JSON document")
    parser.add_argument("--kind", choices=("protocol", "corpus", "report"), required=True)
    parser.add_argument("--protocol", type=Path, default=root / "bakeoff/protocol-v1.json")
    parser.add_argument("--corpus", type=Path, help="corpus JSON used to resolve report shot ids")
    parser.add_argument("--schema", type=Path, help="override the selected schema path")
    args = parser.parse_args(argv)

    protocol_path = args.protocol
    try:
        protocol = load_json(protocol_path)
        version = protocol.get("schema_version", 1)
        if version not in (1, 2):
            raise ValueError(f"unsupported protocol schema version: {version!r}")
        protocol_schema_path = root / f"bakeoff/protocol-v{version}.schema.json"
        report_schema_path = root / f"bakeoff/report-v{version}.schema.json"
        # v2 deliberately reuses the unchanged v1 corpus contract; keep this derived from the
        # protocol version rather than making report validation depend on a filename convention.
        corpus_schema_path = root / "bakeoff/corpus-v1.schema.json"
        protocol_schema = load_json(protocol_schema_path)
        corpus_schema = load_json(corpus_schema_path)
        report_schema = load_json(report_schema_path)
        # Every document kind depends on the same frozen protocol bundle.  Validate that
        # bundle before branching so corpus/report invocations cannot use a mutated cap,
        # provider, or policy table merely because their document schema is valid.
        validate_protocol_and_schemas(protocol, protocol_schema, corpus_schema, report_schema)
        if args.kind == "protocol":
            document = load_json(args.document)
            schema = load_json(args.schema) if args.schema else protocol_schema
            validate(document, schema)
            validate_protocol_consistency(document, schema)
        elif args.kind == "corpus":
            document = load_json(args.document)
            schema = load_json(args.schema) if args.schema else corpus_schema
            validate_corpus_consistency(document, protocol, schema)
        else:
            if args.corpus is None:
                parser.error("--corpus is required when --kind report")
            document = load_json(args.document)
            schema = load_json(args.schema) if args.schema else report_schema
            corpus = load_json(args.corpus)
            validate_report_consistency(document, protocol, schema, corpus, corpus_schema)
    except (OSError, ValueError, ValidationError) as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        return 1
    print(f"valid {args.kind}: {args.document}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
