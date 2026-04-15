"""
Tests for scoring module — verifies deterministic confidence calculation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scoring import (
    EvidenceWeight,
    calculate_confidence,
    calculate_cluster_confidence,
    ScoringConfig,
    evidence_zero_references,
    evidence_scc_isolated,
    evidence_git_stale,
    evidence_no_test_coverage,
    evidence_import_orphan,
    evidence_llm_assessment,
    evidence_feature_flag_retired,
)


# ─── Evidence Weight Values ───────────────────────────────────────

class TestEvidenceWeights:
    def test_strong_weight(self):
        assert EvidenceWeight.value("strong") == 0.30

    def test_moderate_weight(self):
        assert EvidenceWeight.value("moderate") == 0.15

    def test_weak_weight(self):
        assert EvidenceWeight.value("weak") == 0.05

    def test_unknown_weight(self):
        assert EvidenceWeight.value("invented") == 0.0


# ─── Evidence Generators ──────────────────────────────────────────

class TestEvidenceGenerators:
    def test_zero_references_strong(self):
        e = evidence_zero_references(grep_count=0, total_files=1000)
        assert e["weight"] == "strong"
        assert e["source"] == "ripgrep"

    def test_nonzero_references_weak(self):
        e = evidence_zero_references(grep_count=3, total_files=1000)
        assert e["weight"] == "weak"

    def test_scc_isolated_strong(self):
        e = evidence_scc_isolated(scc_size=3, inbound_from_alive=0)
        assert e["weight"] == "strong"

    def test_scc_not_isolated_moderate(self):
        e = evidence_scc_isolated(scc_size=3, inbound_from_alive=1)
        assert e["weight"] == "moderate"

    def test_git_stale_365_strong(self):
        e = evidence_git_stale("2024-01-01", 400)
        assert e["weight"] == "strong"

    def test_git_stale_200_moderate(self):
        e = evidence_git_stale("2025-06-01", 200)
        assert e["weight"] == "moderate"

    def test_git_stale_30_weak(self):
        e = evidence_git_stale("2026-03-01", 30)
        assert e["weight"] == "weak"

    def test_no_tests_moderate(self):
        e = evidence_no_test_coverage(has_tests=False, test_files_checked=50)
        assert e["weight"] == "moderate"

    def test_has_tests_weak(self):
        e = evidence_no_test_coverage(has_tests=True, test_files_checked=50)
        assert e["weight"] == "weak"

    def test_import_orphan_strong(self):
        e = evidence_import_orphan(import_count=0)
        assert e["weight"] == "strong"

    def test_llm_always_weak(self):
        e = evidence_llm_assessment("looks dead", 0)
        assert e["weight"] == "weak"
        assert e["source"] == "llm_judge"

    def test_feature_flag_retired(self):
        e = evidence_feature_flag_retired("old-flag", is_retired=True)
        assert e["signal"] == "feature_flag_retired"
        assert e["weight"] == "strong"

    def test_feature_flag_active(self):
        e = evidence_feature_flag_retired("active-flag", is_retired=False)
        assert e["signal"] == "feature_flag_active"


# ─── Confidence Calculation ───────────────────────────────────────

class TestConfidenceCalculation:
    def test_strong_signals_high_confidence(self):
        evidence = [
            evidence_zero_references(0, 1000),          # strong: 0.30
            evidence_scc_isolated(1, 0),                 # strong: 0.30
            evidence_import_orphan(0),                    # strong: 0.30
            evidence_no_test_coverage(False, 50),         # moderate: 0.15
        ]
        score = calculate_confidence(evidence)
        # Raw: 0.30 + 0.30 + 0.30 + 0.15 = 1.05 / 1.50 = 0.70
        assert score >= 0.65
        assert score <= 0.99

    def test_no_evidence_returns_floor(self):
        score = calculate_confidence([])
        assert score == 0.0

    def test_single_signal_below_minimum(self):
        evidence = [evidence_zero_references(0, 1000)]  # Only 1 signal
        config = ScoringConfig(min_signals=2)
        score = calculate_confidence(evidence, config)
        assert score == 0.0  # Below min_signals

    def test_llm_alive_mechanisms_reduce_score(self):
        evidence = [
            evidence_zero_references(0, 1000),           # strong
            evidence_scc_isolated(1, 0),                  # strong
            evidence_llm_assessment("found reflection", 2),  # 2 alive mechanisms
        ]
        score_with_mechanisms = calculate_confidence(evidence)

        evidence_no_mechanisms = [
            evidence_zero_references(0, 1000),
            evidence_scc_isolated(1, 0),
            evidence_llm_assessment("looks dead", 0),
        ]
        score_without = calculate_confidence(evidence_no_mechanisms)

        # Alive mechanisms should reduce confidence
        assert score_with_mechanisms < score_without

    def test_active_feature_flag_reduces_score(self):
        evidence = [
            evidence_zero_references(0, 1000),
            evidence_scc_isolated(1, 0),
            evidence_feature_flag_retired("flag", is_retired=False),  # ACTIVE = alive
        ]
        score = calculate_confidence(evidence)
        # Active flag subtracts from score
        assert score < 0.5

    def test_ceiling_enforced(self):
        # Even with massive evidence, never exceed ceiling
        evidence = [
            evidence_zero_references(0, 1000),
            evidence_scc_isolated(1, 0),
            evidence_import_orphan(0),
            evidence_git_stale("2020-01-01", 2000),
            evidence_no_test_coverage(False, 100),
        ]
        config = ScoringConfig(confidence_ceiling=0.99)
        score = calculate_confidence(evidence, config)
        assert score <= 0.99


# ─── Cluster Confidence ──────────────────────────────────────────

class TestClusterConfidence:
    def test_min_strategy(self):
        scores = [0.95, 0.88, 0.92]
        result = calculate_cluster_confidence(scores, strategy="min")
        assert result == 0.88

    def test_avg_strategy(self):
        scores = [0.90, 0.80, 0.70]
        result = calculate_cluster_confidence(scores, strategy="avg")
        assert result == pytest.approx(0.80)

    def test_empty_scores(self):
        result = calculate_cluster_confidence([], strategy="min")
        assert result == 0.0

    def test_single_score(self):
        result = calculate_cluster_confidence([0.95], strategy="min")
        assert result == 0.95
