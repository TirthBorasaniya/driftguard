"""Tests for README rendering from results files."""

import subprocess
import sys

import pytest

from driftguard.common.render import render_readme
from driftguard.common.results import write_results
from driftguard.common.settings import REPO_ROOT

pytestmark = pytest.mark.unit

README = """intro
<!-- results:headline:start -->
<!-- results:headline:end -->
<!-- results:v1_verified_run:start -->
stale
<!-- results:v1_verified_run:end -->
<!-- results:unknown_thing:start -->
x
<!-- results:unknown_thing:end -->
"""

V1_METRICS = {
    "split": "s",
    "source_commit": "abc",
    "items": [{"claim": "c", "value": 1, "source": "s"}],
}


def test_missing_results_leave_every_block_empty(tmp_path):
    out = render_readme(README, tmp_path)
    assert out == (
        "intro\n"
        "<!-- results:headline:start -->\n<!-- results:headline:end -->\n"
        "<!-- results:v1_verified_run:start -->\n<!-- results:v1_verified_run:end -->\n"
        "<!-- results:unknown_thing:start -->\n<!-- results:unknown_thing:end -->\n"
    )
    assert "stale" not in out
    assert "not yet run" not in out.lower()


def test_empty_blocks_are_stable_without_results(tmp_path):
    once = render_readme(README, tmp_path)
    assert render_readme(once, tmp_path) == once


def test_render_is_idempotent(tmp_path):
    write_results("v1_verified_run", V1_METRICS, None, notes="note", results_dir=tmp_path)
    once = render_readme(README, tmp_path)
    assert "| c | 1 | s |" in once
    assert render_readme(once, tmp_path) == once


def test_table_is_inserted_only_for_the_matching_results_file(tmp_path):
    write_results("v1_verified_run", V1_METRICS, None, notes="note", results_dir=tmp_path)
    out = render_readme(README, tmp_path)
    assert "| c | 1 | s |" in out
    assert "<!-- results:headline:start -->\n<!-- results:headline:end -->" in out
    assert "<!-- results:unknown_thing:start -->\n<!-- results:unknown_thing:end -->" in out


def test_removed_results_file_clears_its_table(tmp_path):
    write_results("v1_verified_run", V1_METRICS, None, notes="note", results_dir=tmp_path)
    filled = render_readme(README, tmp_path)
    (tmp_path / "v1_verified_run.json").unlink()
    cleared = render_readme(filled, tmp_path)
    assert "| c | 1 | s |" not in cleared
    assert "<!-- results:v1_verified_run:start -->\n<!-- results:v1_verified_run:end -->" in cleared


def test_check_passes_when_no_results_are_present(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(render_readme(README, tmp_path))
    results = tmp_path / "results"
    results.mkdir()
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "render_results.py"),
        "--check",
        "--readme",
        str(readme),
        "--results-dir",
        str(results),
    ]
    done = subprocess.run(command, capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stdout + done.stderr


def test_check_fails_on_a_stale_table(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(README)
    results = tmp_path / "results"
    results.mkdir()
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "render_results.py"),
        "--check",
        "--readme",
        str(readme),
        "--results-dir",
        str(results),
    ]
    assert subprocess.run(command, capture_output=True, text=True, check=False).returncode == 1


def test_committed_readme_has_no_tables_or_placeholders():
    text = (REPO_ROOT / "README.md").read_text()
    assert "not yet run" not in text.lower()
    assert render_readme(text, REPO_ROOT / "does-not-exist") == text
