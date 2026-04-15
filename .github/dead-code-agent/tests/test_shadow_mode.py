"""
Tests for shadow_report and triage_tracker modules.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import DeadSymbol, DeletionType, ZombieCluster
from shadow_report import (
    build_shadow_report,
    save_report,
    load_report,
    load_all_reports,
    report_to_markdown,
    generate_run_id,
)
from triage_tracker import compute_trends, trends_to_markdown, triage_cluster


# ─── Fixtures ──────────────────────────────────────────────────────

def _make_cluster(
    cluster_id: str = "abc123",
    confidence: float = 0.95,
    num_symbols: int = 2,
) -> ZombieCluster:
    symbols = []
    for i in range(num_symbols):
        symbols.append(DeadSymbol(
            fqn=f"src.legacy.func_{i}",
            file_path=f"src/legacy.py",
            start_line=10 + i * 5,
            end_line=14 + i * 5,
            confidence=confidence,
            evidence=[
                {"signal": "zero_references", "weight": "strong", "source": "ripgrep", "detail": "0 matches"},
                {"signal": "scc_isolated", "weight": "strong", "source": "graph", "detail": "SCC of 2"},
            ],
            zombie_cluster_id=cluster_id,
            deletion_type=DeletionType.PURE_DELETION,
        ))
    return ZombieCluster(
        cluster_id=cluster_id,
        symbols=symbols,
        combined_confidence=confidence,
        files_affected=["src/legacy.py"],
    )


# ─── Shadow Report ─────────────────────────────────────────────────

class TestShadowReport:
    def test_builds_report_structure(self):
        clusters = [_make_cluster("abc123", 0.95), _make_cluster("def456", 0.80)]
        passing = [c for c in clusters if c.combined_confidence >= 0.92]

        report = build_shadow_report(
            run_id="20260416-090000",
            all_clusters=clusters,
            threshold_passing=passing,
            roots_by_tier={"tier1_explicit": {"main"}, "tier2_framework": {"view_a", "view_b"}},
            proposed_patterns=[],
            config={"threshold": 0.92},
        )

        assert report["schema_version"] == "1.0"
        assert report["run_id"] == "20260416-090000"
        assert report["metrics"]["total_clusters"] == 2
        assert report["metrics"]["threshold_passing"] == 1
        assert len(report["decisions"]) == 2

    def test_decisions_have_triage_fields(self):
        report = build_shadow_report(
            run_id="test",
            all_clusters=[_make_cluster()],
            threshold_passing=[_make_cluster()],
            roots_by_tier={},
            proposed_patterns=[],
            config={},
        )
        decision = report["decisions"][0]
        assert "triage" in decision
        assert decision["triage"]["status"] == "pending"
        assert decision["triage"]["reviewer"] is None

    def test_passing_vs_below_threshold(self):
        high = _make_cluster("high", 0.95)
        low = _make_cluster("low", 0.80)

        report = build_shadow_report(
            run_id="test",
            all_clusters=[high, low],
            threshold_passing=[high],
            roots_by_tier={},
            proposed_patterns=[],
            config={},
        )

        decisions = {d["cluster_id"]: d for d in report["decisions"]}
        assert decisions["high"]["decision"] == "would_open_pr"
        assert decisions["low"]["decision"] == "below_threshold"

    def test_confidence_histogram(self):
        report = build_shadow_report(
            run_id="test",
            all_clusters=[_make_cluster("a", 0.95), _make_cluster("b", 0.60)],
            threshold_passing=[],
            roots_by_tier={},
            proposed_patterns=[],
            config={},
        )
        hist = report["metrics"]["confidence_distribution"]
        assert hist["0.92-0.95"] == 1
        assert hist["0.50-0.69"] == 1

    def test_root_detection_summary(self):
        report = build_shadow_report(
            run_id="test",
            all_clusters=[],
            threshold_passing=[],
            roots_by_tier={
                "tier1_explicit": {"main", "cli"},
                "tier2_framework": {"view_a"},
            },
            proposed_patterns=[],
            config={},
        )
        root_summary = report["root_detection"]
        assert root_summary["tier1_explicit"]["count"] == 2
        assert root_summary["tier2_framework"]["count"] == 1
        assert root_summary["total"] == 3

    def test_save_and_load(self, tmp_path):
        report = build_shadow_report(
            run_id="test-save",
            all_clusters=[_make_cluster()],
            threshold_passing=[],
            roots_by_tier={},
            proposed_patterns=[],
            config={},
        )
        path = save_report(report, tmp_path)
        loaded = load_report(path)
        assert loaded["run_id"] == "test-save"
        assert loaded["metrics"]["total_clusters"] == 1

    def test_load_all_reports_sorted(self, tmp_path):
        for run_id in ["20260414", "20260416", "20260415"]:
            report = build_shadow_report(
                run_id=run_id,
                all_clusters=[],
                threshold_passing=[],
                roots_by_tier={},
                proposed_patterns=[],
                config={},
            )
            save_report(report, tmp_path)
        reports = load_all_reports(tmp_path)
        assert len(reports) == 3
        # Sorted by filename (oldest first)
        assert reports[0]["run_id"] == "20260414"
        assert reports[2]["run_id"] == "20260416"


class TestReportToMarkdown:
    def test_contains_headline_metrics(self):
        report = build_shadow_report(
            run_id="test",
            all_clusters=[_make_cluster()],
            threshold_passing=[_make_cluster()],
            roots_by_tier={},
            proposed_patterns=[],
            config={},
        )
        md = report_to_markdown(report)
        assert "Shadow Mode Report" in md
        assert "Total clusters found" in md
        assert "Confidence Distribution" in md

    def test_contains_triage_instructions(self):
        report = build_shadow_report(
            run_id="test",
            all_clusters=[_make_cluster()],
            threshold_passing=[],
            roots_by_tier={},
            proposed_patterns=[],
            config={},
        )
        md = report_to_markdown(report)
        assert "Triage Instructions" in md
        assert "true_positive" in md


# ─── Triage Tracker ────────────────────────────────────────────────

class TestComputeTrends:
    def test_empty_reports(self):
        trends = compute_trends([])
        assert "error" in trends

    def test_all_pending(self):
        report = build_shadow_report(
            run_id="run1",
            all_clusters=[_make_cluster()],
            threshold_passing=[_make_cluster()],
            roots_by_tier={},
            proposed_patterns=[],
            config={},
        )
        trends = compute_trends([report])
        assert trends["cumulative"]["pending"] == 1
        assert trends["cumulative"]["false_positive_rate"] is None

    def test_triaged_reports(self):
        report = build_shadow_report(
            run_id="run1",
            all_clusters=[_make_cluster("tp"), _make_cluster("fp")],
            threshold_passing=[_make_cluster("tp"), _make_cluster("fp")],
            roots_by_tier={},
            proposed_patterns=[],
            config={},
        )
        # Manually triage
        report["decisions"][0]["triage"]["status"] = "true_positive"
        report["decisions"][1]["triage"]["status"] = "false_positive"
        report["decisions"][1]["triage"]["notes"] = "Called via reflection"
        report["decisions"][1]["triage"]["root_added"] = "src.rpc.handler"

        trends = compute_trends([report])
        c = trends["cumulative"]
        assert c["true_positives"] == 1
        assert c["false_positives"] == 1
        assert c["false_positive_rate"] == 0.5

        # FP root cause tracked
        assert len(trends["fp_root_causes"]) == 1
        assert trends["fp_root_causes"][0]["root_added"] == "src.rpc.handler"

    def test_per_run_tracking(self):
        reports = []
        for run_id in ["run1", "run2"]:
            r = build_shadow_report(
                run_id=run_id,
                all_clusters=[_make_cluster(f"{run_id}_c")],
                threshold_passing=[_make_cluster(f"{run_id}_c")],
                roots_by_tier={},
                proposed_patterns=[],
                config={},
            )
            r["decisions"][0]["triage"]["status"] = "true_positive"
            reports.append(r)

        trends = compute_trends(reports)
        assert len(trends["per_run"]) == 2
        assert trends["cumulative"]["false_positive_rate"] == 0.0

    def test_confidence_calibration(self):
        report = build_shadow_report(
            run_id="run1",
            all_clusters=[_make_cluster("high", 0.95)],
            threshold_passing=[_make_cluster("high", 0.95)],
            roots_by_tier={},
            proposed_patterns=[],
            config={},
        )
        report["decisions"][0]["triage"]["status"] = "true_positive"
        report["decisions"][0]["combined_confidence"] = 0.95

        trends = compute_trends([report])
        cal = trends["confidence_calibration"]
        assert "0.92-0.95" in cal
        assert cal["0.92-0.95"]["accuracy"] == 1.0


class TestTriageCluster:
    def test_updates_report_file(self, tmp_path):
        report = build_shadow_report(
            run_id="test",
            all_clusters=[_make_cluster("abc12345")],
            threshold_passing=[],
            roots_by_tier={},
            proposed_patterns=[],
            config={},
        )
        path = save_report(report, tmp_path)

        triage_cluster(
            report_path=path,
            cluster_id="abc12345",
            status="false_positive",
            reviewer="engineer-1",
            notes="Used via gRPC reflection",
            root_added="src.legacy.func_0",
        )

        updated = load_report(path)
        triage = updated["decisions"][0]["triage"]
        assert triage["status"] == "false_positive"
        assert triage["reviewer"] == "engineer-1"
        assert triage["notes"] == "Used via gRPC reflection"
        assert triage["root_added"] == "src.legacy.func_0"

        # Triage summary recomputed
        assert updated["triage_summary"]["false_positives"] == 1
        assert updated["triage_summary"]["pending"] == 0


class TestTrendsToMarkdown:
    def test_go_no_go_ready(self):
        report = build_shadow_report(
            run_id="run1",
            all_clusters=[_make_cluster()],
            threshold_passing=[_make_cluster()],
            roots_by_tier={},
            proposed_patterns=[],
            config={},
        )
        report["decisions"][0]["triage"]["status"] = "true_positive"
        trends = compute_trends([report])
        md = trends_to_markdown(trends)
        assert "Ready for production" in md

    def test_go_no_go_not_ready(self):
        report = build_shadow_report(
            run_id="run1",
            all_clusters=[_make_cluster("a"), _make_cluster("b")],
            threshold_passing=[_make_cluster("a"), _make_cluster("b")],
            roots_by_tier={},
            proposed_patterns=[],
            config={},
        )
        report["decisions"][0]["triage"]["status"] = "false_positive"
        report["decisions"][1]["triage"]["status"] = "false_positive"
        trends = compute_trends([report])
        md = trends_to_markdown(trends)
        assert "Not ready" in md

    def test_awaiting_triage(self):
        report = build_shadow_report(
            run_id="run1",
            all_clusters=[_make_cluster()],
            threshold_passing=[],
            roots_by_tier={},
            proposed_patterns=[],
            config={},
        )
        trends = compute_trends([report])
        md = trends_to_markdown(trends)
        assert "Awaiting triage" in md
