"""
Slack notification for dead code PRs.

Sends ONE digest message per run — not one message per PR.
Engineers get a single "here's what happened this week" card
with actionable buttons to review each PR.

Design:
  - Opened PRs → green with "Review PR" buttons
  - Blocked for review → amber with reason
  - Failures → red (infrastructure errors only)
  - Empty run → short "nothing found" message
"""

from __future__ import annotations

import json
import logging
import os
import argparse
from pathlib import Path

logger = logging.getLogger(__name__)


def build_slack_payload(
    results: list[dict],
    run_url: str,
    shadow_mode: bool = False,
    shadow_report_path: str = "",
) -> dict:
    """
    Build a Slack Block Kit payload from PR results.
    Returns a dict ready to POST to a webhook.

    In shadow mode, shows what WOULD have happened with triage instructions.
    """
    opened = [r for r in results if r.get("success")]
    blocked = [
        r for r in results
        if not r.get("success") and r.get("requires_human_review")
    ]
    failed = [
        r for r in results
        if not r.get("success") and not r.get("requires_human_review")
    ]
    skipped = [
        r for r in results
        if not r.get("success") and "skipped" in (r.get("error") or "").lower()
    ]

    # Remove skipped from failed count
    failed = [f for f in failed if f not in skipped]

    # Header
    header_text = (
        "🔍 Dead Code Agent — Shadow Mode Run"
        if shadow_mode
        else "🧹 Dead Code Agent — Weekly Run"
    )
    header = {
        "type": "header",
        "text": {
            "type": "plain_text",
            "text": header_text,
        },
    }

    # Summary counts
    summary_parts = []
    if opened:
        summary_parts.append(f"*{len(opened)} PR{'s' if len(opened) != 1 else ''} opened*")
    if blocked:
        summary_parts.append(f"{len(blocked)} held for human review")
    if failed:
        summary_parts.append(f"{len(failed)} failed")
    if not summary_parts:
        summary_parts.append("No clusters found this week")

    summary = {
        "type": "section",
        "text": {
            "type": "mrkdwn",
            "text": (
                " · ".join(summary_parts)
                + f"\n<{run_url}|View full run logs>"
            ),
        },
    }

    divider = {"type": "divider"}
    blocks: list[dict] = [header]

    # Shadow mode banner
    if shadow_mode:
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    "⚠️ *This is a dry run.* No PRs were opened.\n"
                    "The clusters below show what _would_ have been flagged. "
                    "Please triage each one to calibrate the agent."
                ),
            },
        })

    blocks.extend([summary, divider])

    # ── Opened PRs (or would-be PRs in shadow mode) ────────────────
    display_prs = opened if not shadow_mode else [r for r in results if r.get("success") or r.get("confidence", 0) >= 0.92]
    for pr in display_prs:
        symbols_preview = ", ".join(
            f"`{s.split('.')[-1]}`" for s in pr.get("symbols_deleted", [])[:3]
        )
        remaining = len(pr.get("symbols_deleted", [])) - 3
        if remaining > 0:
            symbols_preview += f" +{remaining} more"

        confidence = pr.get("confidence", 0)
        title = pr.get("title", "Dead code removal")

        if shadow_mode:
            # Shadow mode — no PR URL, just show what would happen
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        f"*🔍 {title}*\n"
                        f"Symbols: {symbols_preview}\n"
                        f"Files: {len(pr.get('files_modified', []))} · "
                        f"Confidence: {confidence:.0%}\n"
                        f"_Would have opened a PR_"
                    ),
                },
            })
        else:
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        f"*<{pr['pr_url']}|{title}>*\n"
                        f"Symbols: {symbols_preview}\n"
                        f"Files: {len(pr.get('files_modified', []))} · "
                        f"Confidence: {confidence:.0%}"
                    ),
                },
                "accessory": {
                    "type": "button",
                    "text": {"type": "plain_text", "text": "Review PR"},
                    "url": pr["pr_url"],
                    "style": "primary",
                },
            })

    # ── Blocked (need human review) ─────────────────────────────────
    if blocked:
        blocks.append(divider)
        blocked_lines = []
        for r in blocked:
            branch = r.get("branch_name", "unknown")
            error = r.get("error", "unknown reason")
            # Truncate long errors for Slack
            if len(error) > 150:
                error = error[:147] + "..."
            blocked_lines.append(f"• `{branch}` — {error}")

        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*🔶 Held for human review:*\n" + "\n".join(blocked_lines),
            },
        })

    # ── Failures (infrastructure errors) ────────────────────────────
    if failed:
        blocks.append(divider)
        failed_lines = []
        for r in failed:
            error = r.get("error", "unknown error")
            if len(error) > 150:
                error = error[:147] + "..."
            failed_lines.append(f"• {error}")

        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*🔴 Errors:*\n" + "\n".join(failed_lines),
            },
        })

    # ── Shadow mode triage footer ──────────────────────────────────
    if shadow_mode and shadow_report_path:
        blocks.append(divider)
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    "*📋 Triage these clusters:*\n"
                    "```\n"
                    "# Mark a cluster as a true positive:\n"
                    "python .github/dead-code-agent/triage_tracker.py triage \\"
                    f"\n  --report {shadow_report_path} \\"
                    "\n  --cluster <ID> --status true_positive --reviewer <you>\n"
                    "\n# Or as a false positive (and add the root):\n"
                    "python .github/dead-code-agent/triage_tracker.py triage \\"
                    f"\n  --report {shadow_report_path} \\"
                    "\n  --cluster <ID> --status false_positive --reviewer <you> \\"
                    "\n  --root-added src.module.Symbol\n"
                    "```"
                ),
            },
        })

    # ── Footer context ──────────────────────────────────────────────
    mode_label = "Shadow Mode" if shadow_mode else "v1.0.0"
    blocks.append({
        "type": "context",
        "elements": [
            {
                "type": "mrkdwn",
                "text": (
                    f"Dead Code Removal Agent {mode_label} · "
                    "Confidence threshold: ≥92% · "
                    "Max 5 PRs/run"
                ),
            }
        ],
    })

    return {"blocks": blocks}


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for Slack notification."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Send Slack digest for dead code agent run",
    )
    parser.add_argument("--results-dir", required=True, help="Directory containing PR result JSONs")
    parser.add_argument("--run-url", required=True, help="URL to the GHA run")
    parser.add_argument("--shadow-mode", action="store_true", help="Shadow mode — show dry-run digest")
    parser.add_argument("--shadow-report", default="", help="Path to the shadow report JSON")
    args = parser.parse_args(argv)

    # Load all result files
    results_dir = Path(args.results_dir)
    results: list[dict] = []

    if results_dir.exists():
        for f in sorted(results_dir.glob("*.json")):
            try:
                results.append(json.loads(f.read_text()))
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSON: %s", f)

    if not results:
        logger.info("No results to notify about — sending empty summary")
        # Still send a notification so the team knows the agent ran
        results = []

    # Get webhook URL
    webhook = os.environ.get("SLACK_WEBHOOK", "")
    if not webhook:
        logger.error("SLACK_WEBHOOK environment variable not set")
        return

    payload = build_slack_payload(
        results, args.run_url,
        shadow_mode=args.shadow_mode,
        shadow_report_path=args.shadow_report,
    )

    # Send to Slack
    try:
        import httpx

        resp = httpx.post(webhook, json=payload, timeout=15)
        resp.raise_for_status()
        logger.info("Slack notified — %d results sent", len(results))
    except ImportError:
        # Fallback to urllib if httpx not available
        import urllib.request

        req = urllib.request.Request(
            webhook,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=15)
        logger.info("Slack notified (urllib fallback) — %d results sent", len(results))


if __name__ == "__main__":
    main()
