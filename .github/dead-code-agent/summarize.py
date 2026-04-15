"""
GitHub Step Summary writer.

Outputs a markdown summary table to stdout (piped into $GITHUB_STEP_SUMMARY
by the workflow). This appears on the Actions run page as a rich summary.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from models import load_clusters_from_json

logger = logging.getLogger(__name__)


def generate_summary(clusters_path: str) -> str:
    """Generate a markdown summary table from the clusters JSON."""
    try:
        clusters = load_clusters_from_json(clusters_path)
    except Exception as exc:
        return f"⚠️ Failed to load clusters: {exc}\n"

    if not clusters:
        return "## 🧹 Dead Code Agent — No Clusters Found\n\nNo dead code clusters met the confidence threshold this run.\n"

    lines = [
        "## 🧹 Dead Code Agent — Analysis Summary\n",
        f"**{len(clusters)} cluster(s)** identified for potential removal.\n",
        "| # | Cluster ID | Symbols | Files | Confidence | Status |",
        "|---|---|---|---|---|---|",
    ]

    for i, cluster in enumerate(clusters, 1):
        cid = cluster.cluster_id[:8]
        sym_count = len(cluster.symbols)
        file_count = len(set(s.file_path for s in cluster.symbols))
        confidence = f"{cluster.combined_confidence:.0%}"

        # Status based on confidence
        if cluster.combined_confidence >= 0.95:
            status = "🟢 High confidence"
        elif cluster.combined_confidence >= 0.92:
            status = "🟡 Above threshold"
        else:
            status = "🔴 Below threshold"

        lines.append(f"| {i} | `{cid}` | {sym_count} | {file_count} | {confidence} | {status} |")

    lines.append("")

    # Detailed breakdown
    lines.append("### Symbol Details\n")
    for cluster in clusters:
        lines.append(f"<details>")
        lines.append(f"<summary><b>Cluster {cluster.cluster_id[:8]}</b> — {len(cluster.symbols)} symbols</summary>\n")
        for sym in cluster.symbols:
            lines.append(f"  - `{sym.fqn}` ({sym.file_path}:{sym.start_line}–{sym.end_line}) — {sym.confidence:.0%}")
        lines.append("\n</details>\n")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate step summary for dead code analysis")
    parser.add_argument("--clusters", required=True, help="Path to clusters JSON")
    args = parser.parse_args(argv)

    summary = generate_summary(args.clusters)
    sys.stdout.write(summary)


if __name__ == "__main__":
    main()
