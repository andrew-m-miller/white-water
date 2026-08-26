#!/usr/bin/env python3
"""Guard: every ``tools/p25_X`` module the CI workflow invokes must actually exist.

The p25-5 and p25-6 lanes of ``.github/workflows/ci.yml`` shell out to the shared, candidate-
neutral packaging/legal tooling under ``tools/p25_5/`` (there is intentionally no ``tools/p25_6/``
package; ``scripts/ci-p25-6-qualify.sh`` reuses the P25-5 tooling unchanged).  A stale
``tools/p25_6/legal_review.py``-style path -- as a shell path or a ``tools.p25_6.licenses`` import
-- would fail with file-not-found at the first legal step of a dispatch, long after the workflow
was edited and never in the local suite.

This test extracts every ``tools/p25_X/<file>.py`` module path the workflow references, in either
the ``tools/p25_5/foo.py`` command form or the ``tools.p25_5.foo`` dotted-import form, and asserts
each resolves to a file that exists in the repository.  It catches the whole class, not one typo.
"""

from __future__ import annotations

from pathlib import Path
import re
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"

# Command form:   python3.11 tools/p25_5/licenses.py runtime ...
_PATH_RE = re.compile(r"tools/(p25_\d+)/([A-Za-z0-9_./-]+\.py)")
# Dotted-import form:   from tools.p25_5.licenses import canonical_sha256
_IMPORT_RE = re.compile(r"tools\.(p25_\d+)\.([A-Za-z0-9_]+)")


def _referenced_module_paths() -> set[Path]:
    text = CI_WORKFLOW.read_text(encoding="utf-8")
    paths: set[Path] = set()
    for package, relative in _PATH_RE.findall(text):
        paths.add(REPO_ROOT / "tools" / package / relative)
    for package, module in _IMPORT_RE.findall(text):
        paths.add(REPO_ROOT / "tools" / package / f"{module}.py")
    return paths


class CiModulePathTests(unittest.TestCase):
    def test_workflow_exists(self) -> None:
        self.assertTrue(CI_WORKFLOW.is_file(), CI_WORKFLOW)

    def test_every_referenced_tools_module_exists(self) -> None:
        referenced = _referenced_module_paths()
        # The workflow does invoke the shared tooling, so this must not be vacuously empty.
        self.assertTrue(referenced, "no tools/p25_X module references found in ci.yml")
        missing = sorted(str(path.relative_to(REPO_ROOT)) for path in referenced if not path.is_file())
        self.assertEqual(missing, [], f"ci.yml references non-existent module paths: {missing}")

    def test_no_p25_6_python_package_is_referenced(self) -> None:
        """There is intentionally no tools/p25_6/ package; the p25-6 lane reuses tools/p25_5/."""

        referenced = _referenced_module_paths()
        stray = sorted(
            str(path.relative_to(REPO_ROOT))
            for path in referenced
            if path.parent.name == "p25_6"
        )
        self.assertEqual(
            stray,
            [],
            f"ci.yml points at a nonexistent tools/p25_6/ package (use tools/p25_5/): {stray}",
        )


if __name__ == "__main__":
    unittest.main()
