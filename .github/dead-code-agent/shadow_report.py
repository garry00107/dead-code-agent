"""
Shadow Report — per-run structured audit trail for dry-run calibration.

Every dry run emits a JSON report to `.github/dead-code-agent/reports/`.
Git IS the observability backend — append-only, already trusted, queryable.

Each report captures:
  - What was found (all clusters, not just threshold-passing ones)
  - What decisions were made (flagged vs. skipped, with reasons)
  - Root detection coverage (which tier caught what)
  - LLM token usage and pattern proposals
  - Engineer triage outcomes (added via triage_tracker.py)

The triage_tracker.py module reads across reports to compute trends:
  - False positive rate over time
  - Root coverage gaps
  - Confidence distribution shifts
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from models import ZombieCluster

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Report Data Structure
# ─────────────────────────────────────────────

def generate_run_id() -> str:
    """Deterministic run ID from timestamp."""
    now = datetime.now(timezone.utc)
    return now.strftime("%Y%m%d-%H%M%S")


def build_shadow_report(
    run_id: str,
    all_clusters: list[ZombieCluster],
    threshold_passing: list[ZombieCluster],
    roots_by_tier: dict[str, set[str]],
    proposed_patterns: list[dict],
    config: dict,
    run_url: str = "",
) -> dict:
    """
    Build a structured shadow report for one dry-run execution.

    This is the single source of truth for what the agent decided and why.
    """
    now = datetime.now(timezone.utc).isoformat()

    # Cluster decision records — every cluster, not just passing ones
    passing_ids = {c.cluster_id for c in threshold_passing}
    decisions = []

    for cluster in all_clusters:
        passed = cluster.cluster_id in passing_ids
        decision = {
            "cluster_id": cluster.cluster_id,
            "symbol_count": len(cluster.symbols),
            "files_affected": cluster.files_affected,
            "combined_confidence": round(cluster.combined_confidence, 4),
            "decision": "would_open_pr" if passed else "below_threshold",
            "symbols": [],
            # Triage fields — filled in by engineers
            "triage": {
                "status": "pending",  # pending | true_positive | false_positive | uncertain
                "reviewer": None,
                "reviewed_at": None,
                "notes": None,
                "root_added": None,   # FQN added to roots.yml, if any
            },
        }

        for sym in cluster.symbols:
            decision["symbols"].append({
                "fqn": sym.fqn,
                "file_path": sym.file_path,
                "lines": f"{sym.start_line}-{sym.end_line}",
                "confidence": round(sym.confidence, 4),
                "evidence_summary": _summarize_evidence(sym.evidence),
                "deletion_type": sym.deletion_type.value,
            })

        decisions.append(decision)

    # Root detection summary
    root_summary = {
        tier: {
            "count": len(roots),
            "sample": sorted(list(roots))[:10],  # Cap to avoid bloat
        }
        for tier, roots in roots_by_tier.items()
    }
    root_summary["total"] = sum(len(r) for r in roots_by_tier.values())

    # Build the report
    report = {
        "schema_version": "1.0",
        "run_id": run_id,
        "timestamp": now,
        "run_url": run_url,
        "config": config,

        # Headline metrics
        "metrics": {
            "total_clusters": len(all_clusters),
            "threshold_passing": len(threshold_passing),
            "total_symbols_flagged": sum(len(c.symbols) for c in threshold_passing),
            "total_files_affected": len(set(
                f for c in threshold_passing for f in c.files_affected
            )),
            "avg_confidence": round(
                sum(c.combined_confidence for c in all_clusters) / max(len(all_clusters), 1), 4,
            ),
            "confidence_distribution": _confidence_histogram(all_clusters),
        },

        "root_detection": root_summary,
        "decisions": decisions,
        "proposed_patterns": proposed_patterns,

        # Triage summary — computed across decisions
        "triage_summary": {
            "total": len(decisions),
            "pending": len(decisions),
            "true_positives": 0,
            "false_positives": 0,
            "uncertain": 0,
            "false_positive_rate": None,  # Computed after triage
        },
    }

    return report


def _summarize_evidence(evidence: list[dict]) -> list[dict]:
    """Compact evidence for the report — enough to understand, not verbose."""
    summary = []
    for e in evidence:
        summary.append({
            "signal": e.get("signal", "unknown"),
            "weight": e.get("weight", "unknown"),
            "source": e.get("source", "unknown"),
            "detail": (e.get("detail", ""))[:120],  # Truncate
        })
    return summary


def _confidence_histogram(clusters: list[ZombieCluster]) -> dict[str, int]:
    """Bucket confidence scores into ranges."""
    buckets = {
        "0.00-0.49": 0,
        "0.50-0.69": 0,
        "0.70-0.84": 0,
        "0.85-0.91": 0,
        "0.92-0.95": 0,
        "0.96-0.99": 0,
    }
    for c in clusters:
        score = c.combined_confidence
        if score < 0.50:
            buckets["0.00-0.49"] += 1
        elif score < 0.70:
            buckets["0.50-0.69"] += 1
        elif score < 0.85:
            buckets["0.70-0.84"] += 1
        elif score < 0.92:
            buckets["0.85-0.91"] += 1
        elif score < 0.96:
            buckets["0.92-0.95"] += 1
        else:
            buckets["0.96-0.99"] += 1
    return buckets


# ─────────────────────────────────────────────
# Report I/O
# ─────────────────────────────────────────────

def save_report(report: dict, reports_dir: Optional[Path] = None) -> Path:
    """Save a shadow report to the reports directory."""
    if reports_dir is None:
        reports_dir = Path(".github/dead-code-agent/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    filename = f"run-{report['run_id']}.json"
    path = reports_dir / filename
    path.write_text(json.dumps(report, indent=2, default=str))
    logger.info("Shadow report saved: %s", path)
    return path


def load_report(path: Path) -> dict:
    """Load a single shadow report."""
    return json.loads(path.read_text())


def load_all_reports(reports_dir: Optional[Path] = None) -> list[dict]:
    """Load all shadow reports, sorted by timestamp (oldest first)."""
    if reports_dir is None:
        reports_dir = Path(".github/dead-code-agent/reports")
    if not reports_dir.exists():
        return []

    reports = []
    for f in sorted(reports_dir.glob("run-*.json")):
        try:
            reports.append(load_report(f))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Skipping malformed report %s: %s", f, exc)
    return reports


# ─────────────────────────────────────────────
# Markdown Summary (for GitHub Step Summary / PR comment)
# ─────────────────────────────────────────────

def report_to_markdown(report: dict) -> str:
    """Convert a shadow report to a human-readable markdown summary."""
    lines = []
    metrics = report.get("metrics", {})
    root_detection = report.get("root_detection", {})

    lines.append("# 🔍 Shadow Mode Report")
    lines.append("")
    lines.append(f"**Run ID:** `{report.get('run_id', 'unknown')}`")
    lines.append(f"**Timestamp:** {report.get('timestamp', 'unknown')}")
    lines.append("")

    # Headline metrics
    lines.append("## Metrics")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Total clusters found | {metrics.get('total_clusters', 0)} |")
    lines.append(f"| Above threshold (would PR) | {metrics.get('threshold_passing', 0)} |")
    lines.append(f"| Symbols flagged | {metrics.get('total_symbols_flagged', 0)} |")
    lines.append(f"| Files affected | {metrics.get('total_files_affected', 0)} |")
    lines.append(f"| Average confidence | {metrics.get('avg_confidence', 0):.1%} |")
    lines.append("")

    # Confidence distribution
    hist = metrics.get("confidence_distribution", {})
    if hist:
        lines.append("### Confidence Distribution")
        lines.append("")
        lines.append("| Range | Count |")
        lines.append("|-------|-------|")
        for bucket, count in hist.items():
            bar = "█" * count
            lines.append(f"| {bucket} | {count} {bar} |")
        lines.append("")

    # Root detection
    lines.append("## Root Detection Coverage")
    lines.append("")
    lines.append(f"| Tier | Roots Found |")
    lines.append(f"|------|-------------|")
    for tier, data in root_detection.items():
        if tier == "total":
            continue
        if isinstance(data, dict):
            lines.append(f"| {tier} | {data.get('count', 0)} |")
    total = root_detection.get("total", 0)
    lines.append(f"| **Total** | **{total}** |")
    lines.append("")

    # Decision details
    decisions = report.get("decisions", [])
    if decisions:
        lines.append("## Decisions")
        lines.append("")
        lines.append("| Cluster | Symbols | Confidence | Decision | Triage |")
        lines.append("|---------|---------|------------|----------|--------|")
        for d in decisions:
            cid = d.get("cluster_id", "")[:8]
            sym_count = d.get("symbol_count", 0)
            conf = d.get("combined_confidence", 0)
            decision = d.get("decision", "unknown")
            triage = d.get("triage", {}).get("status", "pending")
            emoji = {
                "would_open_pr": "🟢",
                "below_threshold": "⚪",
            }.get(decision, "❓")
            triage_emoji = {
                "true_positive": "✅",
                "false_positive": "❌",
                "uncertain": "❓",
                "pending": "⏳",
            }.get(triage, "⏳")
            lines.append(
                f"| `{cid}` | {sym_count} | {conf:.1%} | "
                f"{emoji} {decision} | {triage_emoji} {triage} |"
            )
        lines.append("")

    # Pattern proposals
    proposals = report.get("proposed_patterns", [])
    if proposals:
        lines.append("## LLM Pattern Proposals")
        lines.append("")
        for p in proposals:
            lines.append(
                f"- **{p.get('type', '?')}**: `{p.get('match', '?')}` "
                f"({p.get('language', '?')}) — discovered for `{p.get('discovered_for', '?')}`"
            )
        lines.append("")

    # Triage instructions
    lines.append("## Triage Instructions")
    lines.append("")
    lines.append("To triage a cluster, edit the report JSON and update the `triage` field:")
    lines.append("```json")
    lines.append('"triage": {')
    lines.append('  "status": "true_positive",   // or "false_positive" or "uncertain"')
    lines.append('  "reviewer": "your-github-handle",')
    lines.append('  "reviewed_at": "2026-04-16",')
    lines.append('  "notes": "Actually called via gRPC reflection",')
    lines.append('  "root_added": "src.rpc.handlers.OrderHandler"  // if you added to roots.yml')
    lines.append("}")
    lines.append("```")
    lines.append("")
    lines.append("Then run `python .github/dead-code-agent/triage_tracker.py` to update trends.")

    return "\n".join(lines)
