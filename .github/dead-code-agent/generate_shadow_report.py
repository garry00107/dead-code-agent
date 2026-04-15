#!/usr/bin/env python3
"""
Shadow report generator script for GitHub Actions.

Usage in CI:
    python .github/dead-code-agent/generate_shadow_report.py \
        --clusters /tmp/clusters.json \
        --threshold 0.92 \
        --target src \
        --run-url "https://github.com/..."
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from shadow_report import build_shadow_report, save_report, generate_run_id, report_to_markdown
from models import ZombieCluster


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a shadow report from cluster analysis")
    parser.add_argument("--clusters", required=True, help="Path to clusters JSON file")
    parser.add_argument("--threshold", type=float, default=0.92, help="Confidence threshold")
    parser.add_argument("--target", default="src", help="Target directory that was scanned")
    parser.add_argument("--run-url", default="", help="URL to the GitHub Actions run")
    args = parser.parse_args()

    # Load clusters
    clusters_path = Path(args.clusters)
    if not clusters_path.exists():
        print(f"No clusters file found at {clusters_path}, creating empty report")
        clusters = []
    else:
        with open(clusters_path) as f:
            raw = json.load(f)
        # Reconstruct ZombieCluster objects from JSON
        clusters = []
        for item in raw:
            cluster = ZombieCluster.from_dict(item)
            clusters.append(cluster)

    threshold = args.threshold
    passing = [c for c in clusters if c.combined_confidence >= threshold]

    # Build report
    run_id = generate_run_id()
    report = build_shadow_report(
        run_id=run_id,
        all_clusters=clusters,
        threshold_passing=passing,
        roots_by_tier={},
        proposed_patterns=[],
        config={"threshold": threshold, "target": args.target},
        run_url=args.run_url,
    )

    # Save the report
    report_path = save_report(report)
    print(f"Shadow report saved: {report_path}")

    # Write to GITHUB_OUTPUT if available
    github_output = os.environ.get("GITHUB_OUTPUT", "")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"report_path={report_path}\n")

    # Write markdown to GITHUB_STEP_SUMMARY if available
    github_summary = os.environ.get("GITHUB_STEP_SUMMARY", "")
    if github_summary:
        md = report_to_markdown(report)
        with open(github_summary, "a") as f:
            f.write(md)
    else:
        # Print to stdout for local testing
        print(report_to_markdown(report))


if __name__ == "__main__":
    main()
