#!/usr/bin/env python3
"""Guard: the CI workflow must stay dispatchable under GitHub's 25-input hard cap.

``.github/workflows/ci.yml`` is ``workflow_dispatch`` only -- it is the sole channel that
reaches the airgapped Flame/measurement box.  GitHub enforces a hard limit of **25 inputs** for a
``workflow_dispatch`` event, but *only at dispatch time*: ``yaml.safe_load`` parses a 33-input
workflow without complaint and every local gate passes, yet ``gh workflow run ci.yml ...`` is then
refused with ``HTTP 422: you may only define up to 25 inputs for a workflow_dispatch event`` -- for
every lane at once, including the already-qualified p25-5 one.  That is exactly the regression this
test exists to make un-regressable: the p25-6 packaging work pushed the count to 33 and nothing
caught it because nothing counted the inputs.

The count is kept comfortably below the cap (constants that never vary per dispatch -- conda
installer/lock URLs and SHAs, evaluator/packager/driver entrypoints, package specs, run
instructions, checked-in licence declarations -- are sourced inside the jobs, not as inputs).  Only
the shared ``purpose``/``lane`` and the per-run legal-review attestations the operator supplies
remain inputs.

The parse is dependency-free (no PyYAML, so it runs in the ordinary suite): it reads the single
``on: workflow_dispatch: inputs:`` mapping and counts its direct keys by indentation.
"""

from __future__ import annotations

from pathlib import Path
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"

# GitHub's documented hard cap for a workflow_dispatch event.
GITHUB_DISPATCH_INPUT_CAP = 25
# Our self-imposed ceiling: stay well under the cap so ordinary additions do not silently
# re-approach it.  Raising this is a deliberate act that should be reviewed against the 25 cap.
MARGIN_CAP = 22

# Inputs that genuinely vary per dispatch and must remain inputs.
EXPECTED_INPUTS = {
    "purpose",
    "lane",
    "p25_5_admission_file",
    "p25_5_legal_review_file",
    "p25_5_legal_review_sha256",
    "p25_5_runtime_legal_review_file",
    "p25_5_runtime_legal_review_sha256",
    "p25_6_admission_file",
    "p25_6_legal_review_file",
    "p25_6_legal_review_sha256",
    "p25_6_runtime_legal_review_file",
    "p25_6_runtime_legal_review_sha256",
}


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _dispatch_input_names() -> list[str]:
    """Return the direct child keys of ``on: workflow_dispatch: inputs:`` in ci.yml.

    Dependency-free: locate the ``inputs:`` mapping under ``workflow_dispatch:`` and collect the
    keys indented exactly one YAML level below it (the input names).  Their per-input attributes
    (``description``/``required``/``type``/``default``/``options``) sit one level deeper and are
    excluded by the indentation test, so no ``description:`` text containing a colon is miscounted.
    """

    lines = CI_WORKFLOW.read_text(encoding="utf-8").splitlines()

    # 1) Find the workflow_dispatch: key (a direct child of the top-level on: mapping).
    dispatch_indent = None
    dispatch_line = None
    for index, line in enumerate(lines):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line.strip() == "workflow_dispatch:":
            dispatch_indent = _indent(line)
            dispatch_line = index
            break
    if dispatch_line is None:
        raise AssertionError("ci.yml has no workflow_dispatch: mapping")

    # 2) Find its inputs: child.
    inputs_indent = None
    inputs_line = None
    for index in range(dispatch_line + 1, len(lines)):
        line = lines[index]
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = _indent(line)
        if indent <= dispatch_indent:
            break  # left the workflow_dispatch block without finding inputs
        if line.strip() == "inputs:" and inputs_indent is None:
            inputs_indent = indent
            inputs_line = index
            break
    if inputs_line is None:
        raise AssertionError("workflow_dispatch has no inputs: mapping")

    # 3) Collect the direct child keys of inputs: (one indent level deeper).
    names: list[str] = []
    key_indent = None
    for index in range(inputs_line + 1, len(lines)):
        line = lines[index]
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = _indent(line)
        if indent <= inputs_indent:
            break  # dedented out of the inputs mapping
        if key_indent is None:
            key_indent = indent
        if indent != key_indent:
            continue  # an attribute of an input, not an input name
        stripped = line.strip()
        if stripped.endswith(":"):
            names.append(stripped[:-1])
    return names


class CiDispatchInputCapTests(unittest.TestCase):
    def test_workflow_exists(self) -> None:
        self.assertTrue(CI_WORKFLOW.is_file(), CI_WORKFLOW)

    def test_parser_is_not_vacuous(self) -> None:
        names = _dispatch_input_names()
        self.assertTrue(names, "no workflow_dispatch inputs parsed from ci.yml")
        # The two lane controls anchor that we parsed the right mapping.
        self.assertIn("purpose", names)
        self.assertIn("lane", names)

    def test_input_names_are_unique(self) -> None:
        names = _dispatch_input_names()
        duplicates = sorted({name for name in names if names.count(name) > 1})
        self.assertEqual(duplicates, [], f"duplicate workflow_dispatch inputs: {duplicates}")

    def test_under_github_dispatch_cap(self) -> None:
        count = len(_dispatch_input_names())
        self.assertLessEqual(
            count,
            GITHUB_DISPATCH_INPUT_CAP,
            f"ci.yml defines {count} workflow_dispatch inputs; GitHub refuses dispatch above "
            f"{GITHUB_DISPATCH_INPUT_CAP} (HTTP 422), making every lane un-dispatchable",
        )

    def test_under_margin_cap(self) -> None:
        count = len(_dispatch_input_names())
        self.assertLessEqual(
            count,
            MARGIN_CAP,
            f"ci.yml defines {count} workflow_dispatch inputs; keep it <= {MARGIN_CAP} for margin "
            f"under GitHub's {GITHUB_DISPATCH_INPUT_CAP} cap. Source fixed repo constants inside "
            f"the job instead of adding a dispatch input, or raise MARGIN_CAP deliberately.",
        )

    def test_expected_inputs_are_exactly_the_defined_set(self) -> None:
        names = set(_dispatch_input_names())
        self.assertEqual(
            names,
            EXPECTED_INPUTS,
            "workflow_dispatch inputs drifted from the intended per-run set; "
            f"unexpected={sorted(names - EXPECTED_INPUTS)} missing={sorted(EXPECTED_INPUTS - names)}",
        )


if __name__ == "__main__":
    unittest.main()
