"""
Tests for PRDescriptionGenerator — the audit trail in every PR.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import DeadSymbol, ZombieCluster
from pr_description import PRDescriptionGenerator


# ─── Fixtures ───────────────────────────────────────────────────────

def _make_symbol(
    fqn: str = "src.payments.OldProcessor",
    file_path: str = "src/payments/old.py",
    start_line: int = 10,
    end_line: int = 50,
    confidence: float = 0.95,
    evidence: list[dict] | None = None,
) -> DeadSymbol:
    if evidence is None:
        evidence = [
            {"signal": "zero_references", "detail": "No imports found", "weight": "strong"},
            {"signal": "git_blame", "detail": "Untouched for 14 months", "weight": "moderate"},
        ]
    return DeadSymbol(
        fqn=fqn,
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
        confidence=confidence,
        evidence=evidence,
    )


def _make_cluster(
    symbols: list[DeadSymbol] | None = None,
    confidence: float = 0.95,
) -> ZombieCluster:
    if symbols is None:
        symbols = [_make_symbol()]
    return ZombieCluster(
        cluster_id="abc123def456",
        symbols=symbols,
        combined_confidence=confidence,
        files_affected=[s.file_path for s in symbols],
    )


# ─── Tests ──────────────────────────────────────────────────────────

class TestTitleGeneration:
    def test_single_symbol_title(self):
        gen = PRDescriptionGenerator()
        cluster = _make_cluster(symbols=[_make_symbol(fqn="src.payments.OldProcessor")])
        result = gen.generate(cluster, [])

        assert "OldProcessor" in result["title"]
        assert "chore(dead-code)" in result["title"]
        assert "95%" in result["title"]

    def test_multi_symbol_title_with_common_prefix(self):
        gen = PRDescriptionGenerator()
        symbols = [
            _make_symbol(fqn="src.payments.OldProcessor"),
            _make_symbol(fqn="src.payments.OldValidator"),
        ]
        cluster = _make_cluster(symbols=symbols)
        result = gen.generate(cluster, [])

        assert "src.payments" in result["title"]
        assert "2 symbols" in result["title"]

    def test_multi_symbol_title_without_common_prefix(self):
        gen = PRDescriptionGenerator()
        symbols = [
            _make_symbol(fqn="src.payments.Old"),
            _make_symbol(fqn="src.users.Legacy"),
        ]
        cluster = _make_cluster(symbols=symbols)
        result = gen.generate(cluster, [])

        # Should fall back to something reasonable
        assert "2 symbols" in result["title"] or "src" in result["title"]


class TestBodyContent:
    def test_body_contains_confidence(self):
        gen = PRDescriptionGenerator()
        cluster = _make_cluster()
        result = gen.generate(cluster, [])

        assert "95%" in result["body"]

    def test_body_contains_symbol_fqns(self):
        gen = PRDescriptionGenerator()
        cluster = _make_cluster()
        result = gen.generate(cluster, [])

        assert "src.payments.OldProcessor" in result["body"]

    def test_body_contains_evidence(self):
        gen = PRDescriptionGenerator()
        cluster = _make_cluster()
        result = gen.generate(cluster, [])

        assert "zero_references" in result["body"]
        assert "STRONG" in result["body"]

    def test_body_contains_revert_instructions(self):
        gen = PRDescriptionGenerator()
        cluster = _make_cluster()
        result = gen.generate(cluster, [])

        assert "git revert" in result["body"]

    def test_body_contains_reviewer_checklist(self):
        gen = PRDescriptionGenerator()
        cluster = _make_cluster()
        result = gen.generate(cluster, [])

        assert "- [ ]" in result["body"]
        assert "runtime reflection" in result["body"]
        assert "external consumers" in result["body"]

    def test_body_contains_cluster_id(self):
        gen = PRDescriptionGenerator()
        cluster = _make_cluster()
        result = gen.generate(cluster, [])

        assert "abc123def456" in result["body"]


class TestSafetySummary:
    def test_feature_flag_evidence_reflected(self):
        gen = PRDescriptionGenerator()
        sym = _make_symbol(evidence=[
            {"signal": "feature_flag_retired", "detail": "Flag removed 6 months ago", "weight": "strong"},
        ])
        cluster = _make_cluster(symbols=[sym])
        result = gen.generate(cluster, [])

        assert "flags confirmed retired" in result["body"].lower()

    def test_no_flag_evidence_says_no_dependency(self):
        gen = PRDescriptionGenerator()
        sym = _make_symbol(evidence=[
            {"signal": "zero_references", "detail": "No refs", "weight": "strong"},
        ])
        cluster = _make_cluster(symbols=[sym])
        result = gen.generate(cluster, [])

        assert "No feature flag dependency" in result["body"]


class TestDeletionPlan:
    def test_deletion_steps_table_in_body(self):
        gen = PRDescriptionGenerator()
        cluster = _make_cluster()
        steps = [
            {"action": "delete_file", "file": "src/old.py", "reason": "fully dead"},
            {"action": "remove_lines", "file": "src/util.py", "start_line": 10, "end_line": 20, "symbol": "old_helper"},
        ]
        result = gen.generate(cluster, steps)

        assert "delete_file" in result["body"]
        assert "remove_lines" in result["body"]
        assert "src/old.py" in result["body"]
