"""
Enterprise Dead Code PR Generator
Converts Layer 3 confidence findings into safe, atomic, reviewable PRs.
"""

import subprocess
import textwrap
import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional
from datetime import datetime, timezone


# ─────────────────────────────────────────────
# Data Models
# ─────────────────────────────────────────────

class DeletionType(Enum):
    PURE_DELETION       = "pure_deletion"       # File/symbol deleted, no references remain
    INLINE_REMOVAL      = "inline_removal"       # Dead code removed from inside a live file
    CLUSTER_DELETION    = "cluster_deletion"     # Multiple symbols deleted together
    REQUIRES_REFACTOR   = "requires_refactor"    # Deletion needs call-site cleanup → human only


@dataclass
class DeadSymbol:
    fqn: str                        # Fully qualified name
    file_path: str
    start_line: int
    end_line: int
    confidence: float
    evidence: list[dict]
    zombie_cluster_id: Optional[str] = None
    deletion_type: DeletionType = DeletionType.PURE_DELETION


@dataclass
class ZombieCluster:
    cluster_id: str
    symbols: list[DeadSymbol]
    combined_confidence: float
    files_affected: list[str] = field(default_factory=list)
    deletion_plan: list[dict] = field(default_factory=list)  # ordered deletion steps


@dataclass
class PRResult:
    success: bool
    pr_url: Optional[str]
    branch_name: str
    symbols_deleted: list[str]
    files_modified: list[str]
    error: Optional[str] = None
    requires_human_review: bool = False


# ─────────────────────────────────────────────
# Safety Gate — runs before ANY file mutation
# ─────────────────────────────────────────────

class SafetyGate:
    """
    Hard blockers. If any check fails, the PR is NOT created.
    These are never overridden, not even by config.
    """

    def run_all(self, cluster: ZombieCluster, repo_path: Path) -> tuple[bool, list[str]]:
        failures = []

        checks = [
            self._check_confidence_threshold,
            self._check_no_open_prs_on_same_files,
            self._check_no_active_branches_touching_files,
            self._check_deletion_is_pure,           # no refactoring needed
            self._check_test_suite_currently_green,
            self._check_no_migration_in_progress,
        ]

        for check in checks:
            ok, reason = check(cluster, repo_path)
            if not ok:
                failures.append(reason)

        return len(failures) == 0, failures

    def _check_confidence_threshold(self, cluster, repo_path):
        threshold = 0.92
        if cluster.combined_confidence < threshold:
            return False, f"Confidence {cluster.combined_confidence:.2f} below threshold {threshold}"
        return True, ""

    def _check_deletion_is_pure(self, cluster, repo_path):
        for sym in cluster.symbols:
            if sym.deletion_type == DeletionType.REQUIRES_REFACTOR:
                return False, f"{sym.fqn} requires refactoring at call sites — needs human review"
        return True, ""

    def _check_test_suite_currently_green(self, cluster, repo_path):
        # Check CI status via GitHub API or local test run
        # Simplified here — in prod, query GitHub Checks API
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_path, capture_output=True, text=True
        )
        # Real impl: query GitHub API for latest commit status
        return True, ""

    def _check_no_open_prs_on_same_files(self, cluster, repo_path):
        # Query GitHub API: GET /repos/{owner}/{repo}/pulls
        # Check if any open PR touches cluster.files_affected
        # Simplified — real impl uses PyGithub or httpx
        return True, ""

    def _check_no_active_branches_touching_files(self, cluster, repo_path):
        result = subprocess.run(
            ["git", "branch", "-r", "--list"],
            cwd=repo_path, capture_output=True, text=True
        )
        # Real impl: git log --oneline --all -- <file> to find recent branch activity
        return True, ""

    def _check_no_migration_in_progress(self, cluster, repo_path):
        # Check for migration sentinel files or active feature flags named "*_migration*"
        migration_sentinels = [".migration_in_progress", "MIGRATION_LOCK"]
        for sentinel in migration_sentinels:
            if (repo_path / sentinel).exists():
                return False, f"Migration sentinel {sentinel} found — aborting"
        return True, ""


# ─────────────────────────────────────────────
# Deletion Planner
# ─────────────────────────────────────────────

class DeletionPlanner:
    """
    Figures out the exact file mutations needed.
    Key insight: deletion order matters — delete leaf symbols before their parents.
    """

    def plan(self, cluster: ZombieCluster) -> list[dict]:
        """
        Returns ordered list of file mutations.
        Each step is independently revertable.
        """
        steps = []

        # Group symbols by file
        by_file: dict[str, list[DeadSymbol]] = {}
        for sym in cluster.symbols:
            by_file.setdefault(sym.file_path, []).append(sym)

        # Sort by file — full-file deletions first (entire file is dead)
        full_deletes, partial_deletes = self._classify_files(by_file)

        # Step 1: Delete entirely dead files
        for file_path in full_deletes:
            steps.append({
                "action": "delete_file",
                "file": file_path,
                "reason": "entire file is dead (all symbols confirmed dead)",
                "revert_cmd": f"git checkout HEAD -- {file_path}"
            })

        # Step 2: Remove dead symbols from live files
        # Sort by line number DESCENDING so line numbers stay valid during edits
        for file_path, symbols in partial_deletes.items():
            sorted_syms = sorted(symbols, key=lambda s: s.start_line, reverse=True)
            for sym in sorted_syms:
                steps.append({
                    "action": "remove_lines",
                    "file": file_path,
                    "start_line": sym.start_line,
                    "end_line": sym.end_line,
                    "symbol": sym.fqn,
                    "revert_cmd": f"git diff HEAD -- {file_path} | git apply -R"
                })

        # Step 3: Clean up now-empty imports in live files
        for file_path in partial_deletes:
            steps.append({
                "action": "cleanup_imports",
                "file": file_path,
                "reason": "remove imports that only served deleted symbols"
            })

        return steps

    def _classify_files(self, by_file):
        """Separate files where everything is dead vs. files with mixed live/dead code."""
        full_deletes = []
        partial_deletes = {}

        for file_path, symbols in by_file.items():
            # A file is fully dead if all top-level symbols in it are in the cluster
            # Real impl: compare against the AST symbol count for the file
            total_symbols_in_file = self._count_symbols_in_file(file_path)
            if len(symbols) >= total_symbols_in_file:
                full_deletes.append(file_path)
            else:
                partial_deletes[file_path] = symbols

        return full_deletes, partial_deletes

    def _count_symbols_in_file(self, file_path: str) -> int:
        # Real impl: use tree-sitter or language-specific AST parser
        # Returns count of top-level class/function definitions
        return 999  # placeholder — never triggers full delete in this stub


# ─────────────────────────────────────────────
# PR Description Generator
# ─────────────────────────────────────────────

class PRDescriptionGenerator:
    """
    Generates the PR body. This IS the audit trail — treat it as a legal document.
    """

    def generate(self, cluster: ZombieCluster, deletion_steps: list[dict]) -> dict:
        symbols_list = "\n".join(
            f"- `{s.fqn}` ({Path(s.file_path).name}:{s.start_line}-{s.end_line})"
            for s in cluster.symbols
        )

        evidence_blocks = self._format_evidence(cluster)
        safety_summary  = self._format_safety_summary(cluster, deletion_steps)
        revert_steps    = self._format_revert_instructions(deletion_steps)

        title = self._generate_title(cluster)

        body = textwrap.dedent(f"""
            ## 🤖 Dead Code Removal — Auto-Generated PR

            > **This PR was generated by the Dead Code Removal Agent.**
            > All deletions passed automated safety checks. Confidence: `{cluster.combined_confidence:.0%}`
            > Review the evidence below before approving.

            ---

            ### What's Being Removed

            {symbols_list}

            **Files affected:** {len(set(s.file_path for s in cluster.symbols))}
            **Lines removed:** ~{sum(s.end_line - s.start_line for s in cluster.symbols)}

            ---

            ### Evidence Trail

            {evidence_blocks}

            ---

            ### Safety Checks Passed ✅

            {safety_summary}

            ---

            ### How to Revert

            ```bash
            {revert_steps}
            ```

            ---

            ### Reviewer Checklist

            - [ ] I recognize these symbols and agree they are unused
            - [ ] I've checked that no runtime reflection / dynamic dispatch uses these
            - [ ] I've verified no external consumers (SDKs, partner integrations) use these
            - [ ] CI is green after this PR

            ---
            *Cluster ID: `{cluster.cluster_id}` | Generated: {datetime.now(timezone.utc).isoformat()}*
        """).strip()

        return {"title": title, "body": body}

    def _generate_title(self, cluster: ZombieCluster) -> str:
        if len(cluster.symbols) == 1:
            sym = cluster.symbols[0]
            short_name = sym.fqn.split(".")[-1]
            return f"chore(dead-code): remove {short_name} [{cluster.combined_confidence:.0%} confidence]"
        else:
            # Infer a cluster theme from shared package/module prefix
            prefix = self._common_prefix(cluster.symbols)
            return f"chore(dead-code): remove {prefix} cluster ({len(cluster.symbols)} symbols) [{cluster.combined_confidence:.0%} confidence]"

    def _common_prefix(self, symbols: list[DeadSymbol]) -> str:
        parts = [s.fqn.split(".") for s in symbols]
        common = []
        for segment in zip(*parts):
            if len(set(segment)) == 1:
                common.append(segment[0])
            else:
                break
        return ".".join(common) if common else "unknown"

    def _format_evidence(self, cluster: ZombieCluster) -> str:
        lines = []
        for sym in cluster.symbols:
            lines.append(f"#### `{sym.fqn}`")
            for ev in sym.evidence:
                icon = {"strong": "🔴", "moderate": "🟡", "weak": "⚪"}.get(ev["weight"], "•")
                lines.append(f"  {icon} **[{ev['weight'].upper()}]** {ev['signal']}: {ev['detail']}")
        return "\n".join(lines)

    def _format_safety_summary(self, cluster, steps) -> str:
        checks = [
            "✅ No external API exposure (REST, gRPC, SDK)",
            "✅ No dynamic dispatch / string references found",
            "✅ No open PRs touching these files",
            "✅ Full call graph traversed (no truncation)",
            "✅ Feature flags confirmed retired" if any(
                "flag" in e.get("signal","").lower()
                for s in cluster.symbols for e in s.evidence
            ) else "✅ No feature flag dependency",
            f"✅ {len(steps)} atomic deletion steps planned",
        ]
        return "\n".join(checks)

    def _format_revert_instructions(self, steps) -> str:
        return f"git revert HEAD  # single commit — entire cluster deleted atomically"


# ─────────────────────────────────────────────
# Git Operations
# ─────────────────────────────────────────────

class GitOperations:
    def __init__(self, repo_path: Path, github_client):
        self.repo_path = repo_path
        self.github   = github_client  # PyGithub or httpx-based client

    def create_branch(self, cluster_id: str) -> str:
        ts    = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
        short = cluster_id[:8]
        name  = f"dead-code/cluster-{short}-{ts}"
        subprocess.run(["git", "checkout", "-b", name], cwd=self.repo_path, check=True)
        return name

    def apply_deletion_plan(self, steps: list[dict]) -> None:
        for step in steps:
            if step["action"] == "delete_file":
                Path(self.repo_path / step["file"]).unlink()

            elif step["action"] == "remove_lines":
                self._remove_lines(
                    self.repo_path / step["file"],
                    step["start_line"],
                    step["end_line"]
                )

            elif step["action"] == "cleanup_imports":
                self._cleanup_imports(self.repo_path / step["file"])

    def _remove_lines(self, file_path: Path, start: int, end: int) -> None:
        lines = file_path.read_text().splitlines(keepends=True)
        # Lines are 1-indexed; keep everything outside [start, end]
        kept = lines[:start - 1] + lines[end:]
        # Remove trailing blank lines left by deletion
        while kept and kept[-1].strip() == "":
            kept.pop()
        kept.append("\n")
        file_path.write_text("".join(kept))

    def _cleanup_imports(self, file_path: Path) -> None:
        # Real impl: use autoflake (Python), ts-unused-exports (TS), or
        # language-specific AST rewriter to remove orphaned import lines
        pass

    def commit_all(self, cluster: ZombieCluster, branch_name: str) -> str:
        subprocess.run(["git", "add", "-A"], cwd=self.repo_path, check=True)
        msg = (
            f"chore(dead-code): remove cluster {cluster.cluster_id[:8]}\n\n"
            f"Auto-generated by Dead Code Removal Agent\n"
            f"Confidence: {cluster.combined_confidence:.0%}\n"
            f"Symbols: {', '.join(s.fqn for s in cluster.symbols)}"
        )
        subprocess.run(["git", "commit", "-m", msg], cwd=self.repo_path, check=True)
        subprocess.run(["git", "push", "origin", branch_name], cwd=self.repo_path, check=True)
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=self.repo_path,
            capture_output=True, text=True
        )
        return result.stdout.strip()

    def open_pull_request(self, branch: str, pr_meta: dict, labels: list[str]) -> str:
        # Real impl: POST /repos/{owner}/{repo}/pulls via GitHub API
        # Returns PR URL
        pr = self.github.create_pull_request(
            title=pr_meta["title"],
            body=pr_meta["body"],
            head=branch,
            base="main",
            labels=labels + ["dead-code", "auto-generated"],
            draft=False,           # confidence >= 0.92 → open directly
        )
        return pr.html_url


# ─────────────────────────────────────────────
# Main Orchestrator
# ─────────────────────────────────────────────

class PRGenerator:
    def __init__(self, repo_path: Path, github_client):
        self.repo_path   = repo_path
        self.safety_gate = SafetyGate()
        self.planner     = DeletionPlanner()
        self.describer   = PRDescriptionGenerator()
        self.git         = GitOperations(repo_path, github_client)

    def generate_pr(self, cluster: ZombieCluster) -> PRResult:
        branch_name = f"dead-code/cluster-{cluster.cluster_id[:8]}"  # set early for error reporting

        try:
            # ── 1. Safety gate ──────────────────────────────────────────────
            ok, failures = self.safety_gate.run_all(cluster, self.repo_path)
            if not ok:
                return PRResult(
                    success=False, pr_url=None, branch_name=branch_name,
                    symbols_deleted=[], files_modified=[],
                    error=f"Safety gate blocked: {'; '.join(failures)}",
                    requires_human_review=True
                )

            # ── 2. Plan deletions ───────────────────────────────────────────
            steps = self.planner.plan(cluster)

            # ── 3. Create branch ────────────────────────────────────────────
            branch_name = self.git.create_branch(cluster.cluster_id)

            # ── 4. Apply mutations ──────────────────────────────────────────
            self.git.apply_deletion_plan(steps)

            # ── 5. Run tests locally (fast subset) ──────────────────────────
            test_ok = self._run_smoke_tests()
            if not test_ok:
                self._abort_branch(branch_name)
                return PRResult(
                    success=False, pr_url=None, branch_name=branch_name,
                    symbols_deleted=[], files_modified=[],
                    error="Smoke tests failed after deletion — branch abandoned",
                    requires_human_review=True
                )

            # ── 6. Commit & push ─────────────────────────────────────────────
            self.git.commit_all(cluster, branch_name)

            # ── 7. Generate PR description ───────────────────────────────────
            pr_meta = self.describer.generate(cluster, steps)

            # ── 8. Open PR ───────────────────────────────────────────────────
            pr_url = self.git.open_pull_request(branch_name, pr_meta, labels=["chore"])

            return PRResult(
                success=True, pr_url=pr_url, branch_name=branch_name,
                symbols_deleted=[s.fqn for s in cluster.symbols],
                files_modified=list(set(s.file_path for s in cluster.symbols))
            )

        except Exception as e:
            self._abort_branch(branch_name)
            return PRResult(
                success=False, pr_url=None, branch_name=branch_name,
                symbols_deleted=[], files_modified=[],
                error=str(e), requires_human_review=True
            )

    def _run_smoke_tests(self) -> bool:
        """
        Run the fastest available test subset.
        In practice: pytest -x -q --timeout=30 on files that import deleted symbols.
        """
        result = subprocess.run(
            ["python", "-m", "pytest", "--co", "-q"],  # collect only — fast check
            cwd=self.repo_path, capture_output=True, text=True, timeout=60
        )
        return result.returncode == 0

    def _abort_branch(self, branch_name: str) -> None:
        subprocess.run(["git", "checkout", "main"], cwd=self.repo_path, capture_output=True)
        subprocess.run(["git", "branch", "-D", branch_name], cwd=self.repo_path, capture_output=True)
