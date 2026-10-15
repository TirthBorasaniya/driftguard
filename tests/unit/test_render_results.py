"""Tests for README rendering from results files."""

import pytest

from driftguard.common.render import render_readme
from driftguard.common.results import write_results

pytestmark = pytest.mark.unit

README = """intro
<!-- results:v1_verified_run:start -->
stale
<!-- results:v1_verified_run:end -->
<!-- results:unknown_thing:start -->
x
<!-- results:unknown_thing:end -->
"""


def test_missing_results_render_not_yet_run(tmp_path):
    out = render_readme(README, tmp_path)
    assert "Not yet run" in out
    assert "stale" not in out


def test_render_is_idempotent(tmp_path):
    write_results(
        "v1_verified_run",
        {
            "split": "s",
            "source_commit": "abc",
            "items": [{"claim": "c", "value": 1, "source": "s"}],
        },
        None,
        notes="note",
        results_dir=tmp_path,
    )
    once = render_readme(README, tmp_path)
    assert "| c | 1 | s |" in once
    assert render_readme(once, tmp_path) == once
