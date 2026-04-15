"""
Tests for SafetyGate — the hard-blocker checks that prevent bad PRs.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import DeadSymbol, DeletionType, ZombieCluster
from safety_gate import SafetyGate


# ─── Fixtures ───────────────────────────────────────────────────────

def _make_symbol(
    fqn: str = "src.module.MyClass",
    confidence: float = 0.95,
    deletion_type: DeletionType = DeletionType.PURE_DELETION,
) -> DeadSymbol:
    return DeadSymbol(
        fqn=fqn,
        file_path="src/module.py",
        start_line=10,
        end_line=50,
        confidence=confidence,
        evidence=[{"signal": "zero_references", "detail": "No refs", "weight": "strong"}],
        deletion_type=deletion_type,
    )


def _make_cluster(
    confidence: float = 0.95,
    symbols: list[DeadSymbol] | None = None,
) -> ZombieCluster:
    if symbols is None:
        symbols = [_make_symbol(confidence=confidence)]
    return ZombieCluster(
        cluster_id="abc123def456",
        symbols=symbols,
        combined_confidence=confidence,
        files_affected=[s.file_path for s in symbols],
    )


# ─── Tests ──────────────────────────────────────────────────────────

class TestConfidenceThreshold:
    def test_passes_above_threshold(self):
        gate = SafetyGate(confidence_threshold=0.92)
        cluster = _make_cluster(confidence=0.95)

        with tempfile.TemporaryDirectory() as tmpdir:
            ok, failures = gate.run_all(cluster, Path(tmpdir))

        assert ok is True
        assert failures == []

    def test_fails_below_threshold(self):
        gate = SafetyGate(confidence_threshold=0.92)
        cluster = _make_cluster(confidence=0.85)

        with tempfile.TemporaryDirectory() as tmpdir:
            ok, failures = gate.run_all(cluster, Path(tmpdir))

        assert ok is False
        assert any("below threshold" in f.lower() for f in failures)

    def test_passes_at_exact_threshold(self):
        gate = SafetyGate(confidence_threshold=0.92)
        cluster = _make_cluster(confidence=0.92)

        with tempfile.TemporaryDirectory() as tmpdir:
            ok, failures = gate.run_all(cluster, Path(tmpdir))

        assert ok is True

    def test_custom_threshold(self):
        gate = SafetyGate(confidence_threshold=0.99)
        cluster = _make_cluster(confidence=0.95)

        with tempfile.TemporaryDirectory() as tmpdir:
            ok, failures = gate.run_all(cluster, Path(tmpdir))

        assert ok is False


class TestDeletionPurity:
    def test_pure_deletion_passes(self):
        gate = SafetyGate()
        sym = _make_symbol(deletion_type=DeletionType.PURE_DELETION)
        cluster = _make_cluster(symbols=[sym])

        with tempfile.TemporaryDirectory() as tmpdir:
            ok, failures = gate.run_all(cluster, Path(tmpdir))

        assert ok is True

    def test_requires_refactor_fails(self):
        gate = SafetyGate()
        sym = _make_symbol(deletion_type=DeletionType.REQUIRES_REFACTOR)
        cluster = _make_cluster(symbols=[sym])

        with tempfile.TemporaryDirectory() as tmpdir:
            ok, failures = gate.run_all(cluster, Path(tmpdir))

        assert ok is False
        assert any("refactor" in f.lower() for f in failures)

    def test_mixed_cluster_fails_on_refactor(self):
        gate = SafetyGate()
        symbols = [
            _make_symbol(fqn="src.a.Clean", deletion_type=DeletionType.PURE_DELETION),
            _make_symbol(fqn="src.b.Messy", deletion_type=DeletionType.REQUIRES_REFACTOR),
        ]
        cluster = _make_cluster(symbols=symbols)

        with tempfile.TemporaryDirectory() as tmpdir:
            ok, failures = gate.run_all(cluster, Path(tmpdir))

        assert ok is False


class TestMigrationSentinel:
    def test_passes_without_sentinel(self):
        gate = SafetyGate()
        cluster = _make_cluster()

        with tempfile.TemporaryDirectory() as tmpdir:
            ok, failures = gate.run_all(cluster, Path(tmpdir))

        assert ok is True

    def test_fails_with_migration_lock(self):
        gate = SafetyGate()
        cluster = _make_cluster()

        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "MIGRATION_LOCK").touch()
            ok, failures = gate.run_all(cluster, Path(tmpdir))

        assert ok is False
        assert any("migration" in f.lower() for f in failures)

    def test_fails_with_migration_in_progress(self):
        gate = SafetyGate()
        cluster = _make_cluster()

        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / ".migration_in_progress").touch()
            ok, failures = gate.run_all(cluster, Path(tmpdir))

        assert ok is False


class TestMultipleFailures:
    def test_collects_all_failures(self):
        gate = SafetyGate(confidence_threshold=0.99)
        sym = _make_symbol(confidence=0.80, deletion_type=DeletionType.REQUIRES_REFACTOR)
        cluster = _make_cluster(confidence=0.80, symbols=[sym])

        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "MIGRATION_LOCK").touch()
            ok, failures = gate.run_all(cluster, Path(tmpdir))

        assert ok is False
        # Should have at least 3 failures: confidence, refactor, migration
        assert len(failures) >= 3
