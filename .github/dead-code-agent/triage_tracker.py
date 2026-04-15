"""
Triage Tracker — computes false-positive trends across shadow run reports.

Reads all reports in `.github/dead-code-agent/reports/` and produces:
  1. False-positive rate over time (the most important number)
  2. Root coverage gap analysis (which patterns keep causing FPs)
  3. Confidence calibration curves (are high-confidence decisions trustworthy?)
  4. LLM utility metrics (token spend vs. pattern discovery yield)

This is the "dashboard" — but it runs as a single CLI command and outputs
a markdown summary, no infrastructure required.

Usage:
    python triage_tracker.py --reports-dir .github/dead-code-agent/reports
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from shadow_report import load_all_reports

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Trend Analysis
# ─────────────────────────────────────────────

def compute_trends(reports: list[dict]) -> dict:
    """
    Compute trends across all historical shadow reports.

    Returns a structured summary with:
      - per_run:           array of per-run metrics (for plotting)
      - cumulative:        lifetime totals
      - fp_root_causes:    which FP symbols were added to roots.yml
      - pattern_yield:     confirmed vs. rejected pattern proposals
      - confidence_calibration: accuracy at each confidence bucket
    """
    if not reports:
        return {"error": "No reports found"}

    per_run = []
    cumulative = {
        "total_decisions": 0,
        "triaged": 0,
        "true_positives": 0,
        "false_positives": 0,
        "uncertain": 0,
        "pending": 0,
    }
    fp_root_causes: list[dict] = []
    all_pattern_proposals: list[dict] = []

    # Confidence calibration: bucket → {total, correct}
    calibration: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0})

    for report in reports:
        run_id = report.get("run_id", "unknown")
        decisions = report.get("decisions", [])

        run_stats = {
            "run_id": run_id,
            "timestamp": report.get("timestamp", ""),
            "total_clusters": len(decisions),
            "flagged": sum(1 for d in decisions if d.get("decision") == "would_open_pr"),
            "true_positives": 0,
            "false_positives": 0,
            "uncertain": 0,
            "pending": 0,
            "false_positive_rate": None,
        }

        for d in decisions:
            triage = d.get("triage", {})
            status = triage.get("status", "pending")
            confidence = d.get("combined_confidence", 0)
            bucket = _confidence_to_bucket(confidence)

            cumulative["total_decisions"] += 1

            if status == "true_positive":
                cumulative["true_positives"] += 1
                cumulative["triaged"] += 1
                run_stats["true_positives"] += 1
                calibration[bucket]["total"] += 1
                calibration[bucket]["correct"] += 1

            elif status == "false_positive":
                cumulative["false_positives"] += 1
                cumulative["triaged"] += 1
                run_stats["false_positives"] += 1
                calibration[bucket]["total"] += 1
                # NOT correct — calibration tracks this

                # Collect root cause
                root_added = triage.get("root_added")
                fp_root_causes.append({
                    "run_id": run_id,
                    "cluster_id": d.get("cluster_id", ""),
                    "symbols": [s.get("fqn", "") for s in d.get("symbols", [])],
                    "notes": triage.get("notes", ""),
                    "root_added": root_added,
                })

            elif status == "uncertain":
                cumulative["uncertain"] += 1
                cumulative["triaged"] += 1
                run_stats["uncertain"] += 1

            else:
                cumulative["pending"] += 1
                run_stats["pending"] += 1

        # Per-run false-positive rate
        triaged_this_run = run_stats["true_positives"] + run_stats["false_positives"]
        if triaged_this_run > 0:
            run_stats["false_positive_rate"] = round(
                run_stats["false_positives"] / triaged_this_run, 4,
            )

        per_run.append(run_stats)

        # Collect pattern proposals
        all_pattern_proposals.extend(report.get("proposed_patterns", []))

    # Cumulative false-positive rate
    triaged_total = cumulative["true_positives"] + cumulative["false_positives"]
    cumulative["false_positive_rate"] = (
        round(cumulative["false_positives"] / triaged_total, 4)
        if triaged_total > 0 else None
    )

    # Confidence calibration
    calibration_summary = {}
    for bucket, stats in sorted(calibration.items()):
        if stats["total"] > 0:
            calibration_summary[bucket] = {
                "total": stats["total"],
                "correct": stats["correct"],
                "accuracy": round(stats["correct"] / stats["total"], 4),
            }

    # Pattern yield
    pattern_matches = defaultdict(int)
    for p in all_pattern_proposals:
        key = f"{p.get('type', '?')}:{p.get('match', '?')}"
        pattern_matches[key] += 1

    return {
        "per_run": per_run,
        "cumulative": cumulative,
        "fp_root_causes": fp_root_causes,
        "pattern_proposals": dict(pattern_matches),
        "confidence_calibration": calibration_summary,
    }


def _confidence_to_bucket(confidence: float) -> str:
    """Map a confidence score to a bucket label."""
    if confidence < 0.50:
        return "0.00-0.49"
    elif confidence < 0.70:
        return "0.50-0.69"
    elif confidence < 0.85:
        return "0.70-0.84"
    elif confidence < 0.92:
        return "0.85-0.91"
    elif confidence < 0.96:
        return "0.92-0.95"
    else:
        return "0.96-0.99"


# ─────────────────────────────────────────────
# Markdown Output
# ─────────────────────────────────────────────

def trends_to_markdown(trends: dict) -> str:
    """Convert computed trends to a markdown dashboard."""
    lines = []

    if "error" in trends:
        return f"# Shadow Mode Dashboard\n\nNo reports found. Run the agent in dry-run mode first."

    cumulative = trends.get("cumulative", {})
    per_run = trends.get("per_run", [])

    lines.append("# 📊 Shadow Mode — Triage Dashboard")
    lines.append("")

    # ── Headline Numbers ───────────────────────────────
    fp_rate = cumulative.get("false_positive_rate")
    lines.append("## Headline Metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Total runs | {len(per_run)} |")
    lines.append(f"| Total decisions | {cumulative.get('total_decisions', 0)} |")
    lines.append(f"| Triaged | {cumulative.get('triaged', 0)} |")
    lines.append(f"| ✅ True positives | {cumulative.get('true_positives', 0)} |")
    lines.append(f"| ❌ False positives | {cumulative.get('false_positives', 0)} |")
    lines.append(f"| ❓ Uncertain | {cumulative.get('uncertain', 0)} |")
    lines.append(f"| ⏳ Pending triage | {cumulative.get('pending', 0)} |")
    fp_display = f"{fp_rate:.1%}" if fp_rate is not None else "N/A (nothing triaged)"
    lines.append(f"| **False Positive Rate** | **{fp_display}** |")
    lines.append("")

    # ── Go/No-Go Assessment ────────────────────────────
    lines.append("## Go/No-Go Assessment")
    lines.append("")
    if fp_rate is not None:
        if fp_rate == 0:
            lines.append("> ✅ **Ready for production.** Zero false positives across all triaged clusters.")
        elif fp_rate < 0.05:
            lines.append("> 🟡 **Nearly ready.** False positive rate is below 5%. Review remaining FPs and add roots.")
        elif fp_rate < 0.15:
            lines.append("> 🟠 **More calibration needed.** False positive rate is between 5-15%. Focus on root coverage gaps.")
        else:
            lines.append(f"> 🔴 **Not ready.** False positive rate is {fp_rate:.1%}. Significant root detection gaps remain.")
    else:
        lines.append("> ⏳ **Awaiting triage.** No decisions have been triaged yet. Edit the report JSONs to begin.")
    lines.append("")

    # ── Per-Run Trend ──────────────────────────────────
    if len(per_run) > 1:
        lines.append("## False Positive Rate Over Time")
        lines.append("")
        lines.append("| Run | Date | Flagged | TP | FP | FP Rate |")
        lines.append("|-----|------|---------|----|----|---------|")
        for run in per_run:
            date = run.get("timestamp", "")[:10]
            flagged = run.get("flagged", 0)
            tp = run.get("true_positives", 0)
            fp = run.get("false_positives", 0)
            rate = run.get("false_positive_rate")
            rate_str = f"{rate:.1%}" if rate is not None else "—"
            lines.append(f"| `{run.get('run_id', '?')}` | {date} | {flagged} | {tp} | {fp} | {rate_str} |")
        lines.append("")

    # ── Confidence Calibration ─────────────────────────
    calibration = trends.get("confidence_calibration", {})
    if calibration:
        lines.append("## Confidence Calibration")
        lines.append("")
        lines.append("_Are high-confidence scores actually accurate?_")
        lines.append("")
        lines.append("| Confidence Range | Decisions | Correct | Accuracy |")
        lines.append("|-----------------|-----------|---------|----------|")
        for bucket, stats in calibration.items():
            acc = stats.get("accuracy", 0)
            emoji = "✅" if acc >= 0.95 else "🟡" if acc >= 0.80 else "🔴"
            lines.append(
                f"| {bucket} | {stats['total']} | {stats['correct']} | "
                f"{emoji} {acc:.1%} |"
            )
        lines.append("")

    # ── False Positive Root Causes ─────────────────────
    fp_causes = trends.get("fp_root_causes", [])
    if fp_causes:
        lines.append("## False Positive Root Causes")
        lines.append("")
        lines.append("_These are the FPs that engineers corrected. Each tells you what the detectors missed._")
        lines.append("")
        for fp in fp_causes:
            symbols = ", ".join(f"`{s}`" for s in fp.get("symbols", [])[:3])
            notes = fp.get("notes", "No notes")
            root = fp.get("root_added")
            lines.append(f"- **Cluster `{fp.get('cluster_id', '?')[:8]}`** — {symbols}")
            lines.append(f"  - Reason: {notes}")
            if root:
                lines.append(f"  - Fixed: Added `{root}` to roots.yml")
            lines.append("")

    # ── Pattern Proposal Yield ─────────────────────────
    patterns = trends.get("pattern_proposals", {})
    if patterns:
        lines.append("## LLM Pattern Proposals")
        lines.append("")
        lines.append("| Pattern | Times Proposed |")
        lines.append("|---------|---------------|")
        for pattern, count in sorted(patterns.items(), key=lambda x: -x[1]):
            lines.append(f"| `{pattern}` | {count} |")
        lines.append("")
        lines.append(
            "_Patterns proposed ≥3 times are strong candidates for promotion to "
            "`learned_patterns.yml` with `status: confirmed`._"
        )

    return "\n".join(lines)


# ─────────────────────────────────────────────
# Triage Helper — update report with triage decisions
# ─────────────────────────────────────────────

def triage_cluster(
    report_path: Path,
    cluster_id: str,
    status: str,
    reviewer: str,
    notes: str = "",
    root_added: str = "",
) -> None:
    """
    Update a cluster's triage status in a report file.

    Args:
        report_path: Path to the report JSON
        cluster_id: First 8+ chars of the cluster ID
        status: true_positive | false_positive | uncertain
        reviewer: GitHub handle of the reviewer
        notes: Free-text explanation
        root_added: FQN added to roots.yml (if false positive)
    """
    report = json.loads(report_path.read_text())

    found = False
    for decision in report.get("decisions", []):
        if decision.get("cluster_id", "").startswith(cluster_id):
            decision["triage"] = {
                "status": status,
                "reviewer": reviewer,
                "reviewed_at": datetime.now().strftime("%Y-%m-%d"),
                "notes": notes or None,
                "root_added": root_added or None,
            }
            found = True
            logger.info("Triaged cluster %s as %s", cluster_id, status)
            break

    if not found:
        logger.error("Cluster %s not found in report", cluster_id)
        return

    # Recompute triage summary
    decisions = report.get("decisions", [])
    summary = {
        "total": len(decisions),
        "pending": 0,
        "true_positives": 0,
        "false_positives": 0,
        "uncertain": 0,
        "false_positive_rate": None,
    }
    for d in decisions:
        s = d.get("triage", {}).get("status", "pending")
        if s == "true_positive":
            summary["true_positives"] += 1
        elif s == "false_positive":
            summary["false_positives"] += 1
        elif s == "uncertain":
            summary["uncertain"] += 1
        else:
            summary["pending"] += 1

    triaged = summary["true_positives"] + summary["false_positives"]
    if triaged > 0:
        summary["false_positive_rate"] = round(summary["false_positives"] / triaged, 4)

    report["triage_summary"] = summary
    report_path.write_text(json.dumps(report, indent=2))
    logger.info("Report updated: %s", report_path)


# ─────────────────────────────────────────────
# CLI Entry Points
# ─────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Shadow mode triage tracker — compute trends across dry runs",
    )
    subparsers = parser.add_subparsers(dest="command")

    # ── Dashboard ──────────────────────────────────────
    dash = subparsers.add_parser("dashboard", help="Generate markdown dashboard from all reports")
    dash.add_argument(
        "--reports-dir", default=".github/dead-code-agent/reports",
        help="Directory containing shadow reports",
    )
    dash.add_argument(
        "--output", default=None,
        help="Output file (default: stdout)",
    )

    # ── Triage ─────────────────────────────────────────
    tri = subparsers.add_parser("triage", help="Triage a cluster in a report")
    tri.add_argument("--report", required=True, help="Path to the report JSON")
    tri.add_argument("--cluster", required=True, help="Cluster ID (first 8+ chars)")
    tri.add_argument(
        "--status", required=True,
        choices=["true_positive", "false_positive", "uncertain"],
    )
    tri.add_argument("--reviewer", required=True, help="Your GitHub handle")
    tri.add_argument("--notes", default="", help="Explanation")
    tri.add_argument("--root-added", default="", help="FQN added to roots.yml")

    args = parser.parse_args(argv)

    if args.command == "dashboard":
        reports = load_all_reports(Path(args.reports_dir))
        trends = compute_trends(reports)
        markdown = trends_to_markdown(trends)

        if args.output:
            Path(args.output).write_text(markdown)
            logger.info("Dashboard written to %s", args.output)
        else:
            print(markdown)

    elif args.command == "triage":
        triage_cluster(
            report_path=Path(args.report),
            cluster_id=args.cluster,
            status=args.status,
            reviewer=args.reviewer,
            notes=args.notes,
            root_added=args.root_added,
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
