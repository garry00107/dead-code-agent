"""
Tests for Slack notification payload builder.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from notify_slack import build_slack_payload


# ─── Test Data ──────────────────────────────────────────────────────

SAMPLE_SUCCESS = {
    "success": True,
    "pr_url": "https://github.com/owner/repo/pull/42",
    "branch_name": "dead-code/cluster-abc123",
    "symbols_deleted": ["src.payments.OldProcessor", "src.payments.OldValidator", "src.payments.OldHelper", "src.payments.OldFormatter"],
    "files_modified": ["src/payments/old.py", "src/payments/validators.py"],
    "confidence": 0.95,
    "title": "chore(dead-code): remove payments cluster (4 symbols) [95% confidence]",
    "error": None,
    "requires_human_review": False,
}

SAMPLE_BLOCKED = {
    "success": False,
    "pr_url": None,
    "branch_name": "dead-code/cluster-def456",
    "symbols_deleted": [],
    "files_modified": [],
    "confidence": 0.93,
    "error": "Safety gate blocked: PR #15 touches src/payments/old.py — conflict risk",
    "requires_human_review": True,
}

SAMPLE_FAILED = {
    "success": False,
    "pr_url": None,
    "branch_name": "dead-code/cluster-ghi789",
    "symbols_deleted": [],
    "files_modified": [],
    "confidence": 0.0,
    "error": "Network timeout reaching GitHub API",
    "requires_human_review": False,
}

SAMPLE_SKIPPED = {
    "success": False,
    "pr_url": None,
    "branch_name": "",
    "symbols_deleted": [],
    "files_modified": [],
    "confidence": 0.0,
    "error": "Batch index exceeds cluster count — skipped",
    "requires_human_review": False,
}

RUN_URL = "https://github.com/owner/repo/actions/runs/12345"


# ─── Tests ──────────────────────────────────────────────────────────

class TestPayloadStructure:
    def test_has_blocks(self):
        payload = build_slack_payload([SAMPLE_SUCCESS], RUN_URL)
        assert "blocks" in payload
        assert len(payload["blocks"]) > 0

    def test_starts_with_header(self):
        payload = build_slack_payload([SAMPLE_SUCCESS], RUN_URL)
        assert payload["blocks"][0]["type"] == "header"
        assert "Dead Code Agent" in payload["blocks"][0]["text"]["text"]

    def test_summary_section(self):
        payload = build_slack_payload([SAMPLE_SUCCESS, SAMPLE_BLOCKED], RUN_URL)
        summary = payload["blocks"][1]
        assert summary["type"] == "section"
        assert "1 PR opened" in summary["text"]["text"]
        assert "1 held for human review" in summary["text"]["text"]

    def test_run_url_linked(self):
        payload = build_slack_payload([], RUN_URL)
        summary = payload["blocks"][1]
        assert RUN_URL in summary["text"]["text"]


class TestOpenedPRs:
    def test_pr_button_present(self):
        payload = build_slack_payload([SAMPLE_SUCCESS], RUN_URL)
        pr_blocks = [
            b for b in payload["blocks"]
            if b.get("type") == "section" and "accessory" in b
        ]
        assert len(pr_blocks) == 1
        assert pr_blocks[0]["accessory"]["type"] == "button"
        assert pr_blocks[0]["accessory"]["url"] == SAMPLE_SUCCESS["pr_url"]

    def test_symbols_preview_truncated(self):
        payload = build_slack_payload([SAMPLE_SUCCESS], RUN_URL)
        pr_blocks = [
            b for b in payload["blocks"]
            if b.get("type") == "section" and "accessory" in b
        ]
        text = pr_blocks[0]["text"]["text"]
        # Should show 3 symbols + "+1 more"
        assert "+1 more" in text

    def test_confidence_displayed(self):
        payload = build_slack_payload([SAMPLE_SUCCESS], RUN_URL)
        pr_blocks = [
            b for b in payload["blocks"]
            if b.get("type") == "section" and "accessory" in b
        ]
        text = pr_blocks[0]["text"]["text"]
        assert "95%" in text


class TestBlockedPRs:
    def test_blocked_section_present(self):
        payload = build_slack_payload([SAMPLE_BLOCKED], RUN_URL)
        blocked_blocks = [
            b for b in payload["blocks"]
            if b.get("type") == "section"
            and "🔶" in b.get("text", {}).get("text", "")
        ]
        assert len(blocked_blocks) == 1

    def test_blocked_shows_reason(self):
        payload = build_slack_payload([SAMPLE_BLOCKED], RUN_URL)
        blocked_blocks = [
            b for b in payload["blocks"]
            if b.get("type") == "section"
            and "🔶" in b.get("text", {}).get("text", "")
        ]
        text = blocked_blocks[0]["text"]["text"]
        assert "conflict risk" in text


class TestFailures:
    def test_failed_section_present(self):
        payload = build_slack_payload([SAMPLE_FAILED], RUN_URL)
        error_blocks = [
            b for b in payload["blocks"]
            if b.get("type") == "section"
            and "🔴" in b.get("text", {}).get("text", "")
        ]
        assert len(error_blocks) == 1

    def test_skipped_not_counted_as_failure(self):
        payload = build_slack_payload([SAMPLE_SKIPPED], RUN_URL)
        error_blocks = [
            b for b in payload["blocks"]
            if b.get("type") == "section"
            and "🔴" in b.get("text", {}).get("text", "")
        ]
        # Skipped results should not appear in the error section
        assert len(error_blocks) == 0


class TestEmptyResults:
    def test_empty_results_still_sends(self):
        payload = build_slack_payload([], RUN_URL)
        assert "blocks" in payload
        summary = payload["blocks"][1]
        assert "No clusters found" in summary["text"]["text"]


class TestFooter:
    def test_context_block_present(self):
        payload = build_slack_payload([SAMPLE_SUCCESS], RUN_URL)
        context_blocks = [b for b in payload["blocks"] if b["type"] == "context"]
        assert len(context_blocks) == 1
        assert "v1.0.0" in context_blocks[0]["elements"][0]["text"]


class TestPayloadSerializable:
    def test_payload_is_json_serializable(self):
        """Slack webhook requires JSON-serializable payload."""
        payload = build_slack_payload(
            [SAMPLE_SUCCESS, SAMPLE_BLOCKED, SAMPLE_FAILED, SAMPLE_SKIPPED],
            RUN_URL,
        )
        # Should not raise
        serialized = json.dumps(payload)
        assert len(serialized) > 0

    def test_large_error_truncated(self):
        """Very long error messages should be truncated for Slack."""
        long_error = {
            **SAMPLE_BLOCKED,
            "error": "x" * 500,
        }
        payload = build_slack_payload([long_error], RUN_URL)
        blocked_blocks = [
            b for b in payload["blocks"]
            if b.get("type") == "section"
            and "🔶" in b.get("text", {}).get("text", "")
        ]
        text = blocked_blocks[0]["text"]["text"]
        assert "..." in text
        assert len(text) < 600  # Reasonable Slack block size
