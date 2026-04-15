"""
Tests for DeletionPlanner — ordered mutation planning.
"""

import tempfile
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import DeadSymbol, DeletionType, ZombieCluster
from deletion_planner import DeletionPlanner


# ─── Fixtures ───────────────────────────────────────────────────────

def _make_symbol(
    fqn: str = "src.module.func",
    file_path: str = "src/module.py",
    start_line: int = 10,
    end_line: int = 20,
    confidence: float = 0.95,
) -> DeadSymbol:
    return DeadSymbol(
        fqn=fqn,
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
        confidence=confidence,
        evidence=[],
    )


def _make_cluster(symbols: list[DeadSymbol]) -> ZombieCluster:
    return ZombieCluster(
        cluster_id="test-cluster-001",
        symbols=symbols,
        combined_confidence=min(s.confidence for s in symbols),
        files_affected=list(set(s.file_path for s in symbols)),
    )


def _create_python_file(tmpdir: Path, rel_path: str, content: str) -> Path:
    full = tmpdir / rel_path
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(content)
    return full


# ─── Tests ──────────────────────────────────────────────────────────

class TestFileClassification:
    def test_single_symbol_in_single_function_file(self):
        """A file with one function and one dead symbol → full delete."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            _create_python_file(tmpdir, "src/mod.py", "def only_func():\n    pass\n")

            sym = _make_symbol(fqn="src.mod.only_func", file_path="src/mod.py", start_line=1, end_line=2)
            cluster = _make_cluster([sym])
            planner = DeletionPlanner(repo_path=tmpdir)

            steps = planner.plan(cluster)

        # Should be a full-file delete (1 symbol = 1 function in file)
        delete_steps = [s for s in steps if s["action"] == "delete_file"]
        assert len(delete_steps) == 1
        assert delete_steps[0]["file"] == "src/mod.py"

    def test_partial_delete_when_live_code_remains(self):
        """File has 2 functions, only 1 is dead → partial edit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            content = "def live_func():\n    pass\n\ndef dead_func():\n    pass\n"
            _create_python_file(tmpdir, "src/mod.py", content)

            sym = _make_symbol(fqn="src.mod.dead_func", file_path="src/mod.py", start_line=4, end_line=5)
            cluster = _make_cluster([sym])
            planner = DeletionPlanner(repo_path=tmpdir)

            steps = planner.plan(cluster)

        remove_steps = [s for s in steps if s["action"] == "remove_lines"]
        assert len(remove_steps) == 1
        assert remove_steps[0]["start_line"] == 4
        assert remove_steps[0]["end_line"] == 5


class TestLineOrdering:
    def test_lines_deleted_in_reverse_order(self):
        """Multiple dead symbols in same file → deleted bottom-up."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            content = (
                "def func_a():\n    pass\n\n"
                "def func_b():\n    pass\n\n"
                "def func_c():\n    pass\n"
            )
            _create_python_file(tmpdir, "src/mod.py", content)

            symbols = [
                _make_symbol(fqn="src.mod.func_a", file_path="src/mod.py", start_line=1, end_line=2),
                _make_symbol(fqn="src.mod.func_c", file_path="src/mod.py", start_line=7, end_line=8),
            ]
            cluster = _make_cluster(symbols)
            planner = DeletionPlanner(repo_path=tmpdir)

            steps = planner.plan(cluster)

        remove_steps = [s for s in steps if s["action"] == "remove_lines"]
        assert len(remove_steps) == 2
        # Verify reverse order (higher line numbers first)
        assert remove_steps[0]["start_line"] > remove_steps[1]["start_line"]


class TestImportCleanup:
    def test_import_cleanup_step_added_for_partial_edits(self):
        """Partial edits should have a cleanup_imports step following the removals."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            content = "def live():\n    pass\n\ndef dead():\n    pass\n"
            _create_python_file(tmpdir, "src/mod.py", content)

            sym = _make_symbol(file_path="src/mod.py", start_line=4, end_line=5)
            cluster = _make_cluster([sym])
            planner = DeletionPlanner(repo_path=tmpdir)

            steps = planner.plan(cluster)

        cleanup_steps = [s for s in steps if s["action"] == "cleanup_imports"]
        assert len(cleanup_steps) == 1
        assert cleanup_steps[0]["file"] == "src/mod.py"

    def test_no_import_cleanup_for_full_deletes(self):
        """Full-file deletes don't need import cleanup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            _create_python_file(tmpdir, "src/mod.py", "def only():\n    pass\n")

            sym = _make_symbol(file_path="src/mod.py", start_line=1, end_line=2)
            cluster = _make_cluster([sym])
            planner = DeletionPlanner(repo_path=tmpdir)

            steps = planner.plan(cluster)

        cleanup_steps = [s for s in steps if s["action"] == "cleanup_imports"]
        assert len(cleanup_steps) == 0


class TestRevertCommands:
    def test_full_delete_has_revert_cmd(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            _create_python_file(tmpdir, "src/mod.py", "def func():\n    pass\n")

            sym = _make_symbol(file_path="src/mod.py", start_line=1, end_line=2)
            cluster = _make_cluster([sym])
            planner = DeletionPlanner(repo_path=tmpdir)

            steps = planner.plan(cluster)

        delete_steps = [s for s in steps if s["action"] == "delete_file"]
        assert all("revert_cmd" in s for s in delete_steps)

    def test_line_removal_has_revert_cmd(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            content = "def live():\n    pass\n\ndef dead():\n    pass\n"
            _create_python_file(tmpdir, "src/mod.py", content)

            sym = _make_symbol(file_path="src/mod.py", start_line=4, end_line=5)
            cluster = _make_cluster([sym])
            planner = DeletionPlanner(repo_path=tmpdir)

            steps = planner.plan(cluster)

        remove_steps = [s for s in steps if s["action"] == "remove_lines"]
        assert all("revert_cmd" in s for s in remove_steps)


class TestPlanValidation:
    def test_valid_plan_passes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            content = "def live():\n    pass\n\ndef dead():\n    pass\n"
            _create_python_file(tmpdir, "src/mod.py", content)

            sym = _make_symbol(file_path="src/mod.py", start_line=4, end_line=5)
            cluster = _make_cluster([sym])
            planner = DeletionPlanner(repo_path=tmpdir)

            steps = planner.plan(cluster)
            valid, errors = planner.validate_plan(steps, tmpdir)

        assert valid is True
        assert errors == []

    def test_missing_file_fails_validation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            planner = DeletionPlanner(repo_path=tmpdir)

            steps = [{"action": "delete_file", "file": "nonexistent.py"}]
            valid, errors = planner.validate_plan(steps, tmpdir)

        assert valid is False
        assert len(errors) == 1

    def test_out_of_range_lines_fail_validation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            _create_python_file(tmpdir, "src/mod.py", "line1\nline2\n")
            planner = DeletionPlanner(repo_path=tmpdir)

            steps = [{"action": "remove_lines", "file": "src/mod.py", "start_line": 1, "end_line": 100}]
            valid, errors = planner.validate_plan(steps, tmpdir)

        assert valid is False
