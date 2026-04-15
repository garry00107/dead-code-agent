"""
Slack notification for dead code PRs.
Sends one digest message — not one message per PR.
Engineers get a single "here's what happened this week" card.
"""

import json
import httpx
import argparse
from pathlib import Path


def build_slack_payload(results: list[dict], run_url: str) -> dict:
    opened    = [r for r in results if r.get("success")]
    blocked   = [r for r in results if not r.get("success") and r.get("requires_human_review")]
    failed    = [r for r in results if not r.get("success") and not r.get("requires_human_review")]

    # Header block
    header = {
        "type": "header",
        "text": {
            "type": "plain_text",
            "text": f"🧹 Dead Code Agent — Weekly Run"
        }
    }

    # Summary line
    summary = {
        "type": "section",
        "text": {
            "type": "mrkdwn",
            "text": (
                f"*{len(opened)} PRs opened* · "
                f"{len(blocked)} held for human review · "
                f"{len(failed)} failed\n"
                f"<{run_url}|View full run logs>"
            )
        }
    }

    divider = {"type": "divider"}
    blocks  = [header, summary, divider]

    # One entry per opened PR
    for pr in opened:
        symbols_preview = ", ".join(f"`{s.split('.')[-1]}`" for s in pr["symbols_deleted"][:3])
        if len(pr["symbols_deleted"]) > 3:
            symbols_preview += f" +{len(pr['symbols_deleted']) - 3} more"

        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    f"*<{pr['pr_url']}|{pr['title']}>*\n"
                    f"Symbols: {symbols_preview}\n"
                    f"Files: {len(pr['files_modified'])} · "
                    f"Confidence: {pr['confidence']:.0%}"
                )
            },
            "accessory": {
                "type": "button",
                "text": {"type": "plain_text", "text": "Review PR"},
                "url": pr["pr_url"],
                "style": "primary"
            }
        })

    # Blocked PRs (need human review)
    if blocked:
        blocks.append(divider)
        blocked_text = "\n".join(
            f"• `{r['branch_name']}` — {r['error']}" for r in blocked
        )
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*🔶 Held for human review:*\n{blocked_text}"
            }
        })

    return {"blocks": blocks}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--run-url",     required=True)
    args = parser.parse_args()

    results = []
    for f in Path(args.results_dir).glob("*.json"):
        results.append(json.loads(f.read_text()))

    if not results:
        print("No results to notify about")
        return

    import os
    webhook = os.environ["SLACK_WEBHOOK"]
    payload = build_slack_payload(results, args.run_url)
    resp    = httpx.post(webhook, json=payload)
    resp.raise_for_status()
    print(f"Slack notified — {len(results)} results")


if __name__ == "__main__":
    main()
