"""
Tests for PRGeneratorOrchestrator — end-to-end pipeline with mocked git ops.
"""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import DeadSymbol, DeletionType, ZombieCluster, PRResult
from pr_generator import PRGeneratorOrchestrator, parse_args


# ─── Fixtures ───────────────────────────────────────────────────────

def _make_symbol(
    fqn: str = "src.module.DeadClass",
    file_path: str = "src/module.py",
    start_line: int = 10,
    end_line: int = 30,
    confidence: float = 0.95,
    deletion_type: DeletionType = DeletionType.PURE_DELETION,
) -> DeadSymbol:
    return DeadSymbol(
        fqn=fqn,
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
        confidence=confidence,
        evidence=[{"signal": "zero_references", "detail": "No refs", "weight": "strong"}],
        deletion_type=deletion_type,
    )


def _make_cluster(confidence: float = 0.95) -> ZombieCluster:
    return ZombieCluster(
        cluster_id="abc123def456",
        symbols=[_make_symbol(confidence=confidence)],
        combined_confidence=confidence,
        files_affected=["src/module.py"],
    )


# ─── Tests ──────────────────────────────────────────────────────────

class TestSafetyGateBlocking:
    def test_low_confidence_blocks_pr(self):
        """Clusters below threshold should be blocked, not errored."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orch = PRGeneratorOrchestrator(
                repo_path=Path(tmpdir),
                github_token="fake-token",
                repo_slug="owner/repo",
                confidence_threshold=0.99,
            )

            cluster = _make_cluster(confidence=0.85)
            result = orch.generate_pr(cluster)

        assert result.success is False
        assert result.requires_human_review is True
        assert "Safety gate blocked" in result.error

    def test_refactor_needed_blocks_pr(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sym = _make_symbol(deletion_type=DeletionType.REQUIRES_REFACTOR)
            cluster = ZombieCluster(
                cluster_id="test123",
                symbols=[sym],
                combined_confidence=0.98,
                files_affected=["src/module.py"],
            )

            orch = PRGeneratorOrchestrator(
                repo_path=Path(tmpdir),
                github_token="fake-token",
                repo_slug="owner/repo",
            )
            result = orch.generate_pr(cluster)

        assert result.success is False
        assert "refactor" in result.error.lower()

    def test_migration_lock_blocks_pr(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            (tmpdir_path / "MIGRATION_LOCK").touch()

            orch = PRGeneratorOrchestrator(
                repo_path=tmpdir_path,
                github_token="fake-token",
                repo_slug="owner/repo",
            )

            cluster = _make_cluster()
            result = orch.generate_pr(cluster)

        assert result.success is False
        assert "migration" in result.error.lower()


class TestParseArgs:
    def test_all_required_args(self):
        args = parse_args([
            "--clusters", "/tmp/clusters.json",
            "--batch-index", "2",
            "--repo", "owner/repo",
            "--output", "/tmp/result.json",
        ])
        assert args.clusters == "/tmp/clusters.json"
        assert args.batch_index == 2
        assert args.repo == "owner/repo"
        assert args.output == "/tmp/result.json"
        assert args.base_branch == "main"  # default

    def test_optional_args(self):
        args = parse_args([
            "--clusters", "/tmp/c.json",
            "--batch-index", "0",
            "--repo", "o/r",
            "--output", "/tmp/r.json",
            "--base-branch", "develop",
            "--confidence-threshold", "0.95",
        ])
        assert args.base_branch == "develop"
        assert args.confidence_threshold == 0.95

    def test_missing_required_args_exits(self):
        with pytest.raises(SystemExit):
            parse_args(["--clusters", "/tmp/c.json"])


class TestResultSerialization:
    def test_blocked_result_serializes(self):
        result = PRResult(
            success=False,
            pr_url=None,
            branch_name="dead-code/cluster-abc123",
            symbols_deleted=[],
            files_modified=[],
            confidence=0.85,
            error="Safety gate blocked: confidence 0.85 below threshold 0.92",
            requires_human_review=True,
        )

        d = result.to_dict()
        restored = PRResult.from_dict(d)

        assert restored.success is False
        assert restored.requires_human_review is True
        assert restored.error == result.error

    def test_success_result_serializes(self):
        result = PRResult(
            success=True,
            pr_url="https://github.com/owner/repo/pull/42",
            branch_name="dead-code/cluster-abc123-20260416-0900",
            symbols_deleted=["src.module.DeadClass"],
            files_modified=["src/module.py"],
            confidence=0.95,
            title="chore(dead-code): remove DeadClass [95% confidence]",
        )

        d = result.to_dict()
        restored = PRResult.from_dict(d)

        assert restored.success is True
        assert restored.pr_url == result.pr_url
        assert restored.symbols_deleted == result.symbols_deleted
