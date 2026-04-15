"""
Deletion Planner — figures out the exact file mutations for a zombie cluster.

Key insight: deletion ORDER matters.
  1. Full-file deletes first (simplest, cleanest)
  2. Partial edits in reverse line order (preserves line numbers during edits)
  3. Import cleanup last (removes orphaned imports from edited files)

Every step includes a revert_cmd so the PR description doubles as a runbook.
"""

from __future__ import annotations

import ast
import logging
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Optional

import sys as _sys
from pathlib import Path as _Path
_SCRIPT_DIR = _Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in _sys.path:
    _sys.path.insert(0, str(_SCRIPT_DIR))

from models import DeadSymbol, ZombieCluster

logger = logging.getLogger(__name__)


class DeletionPlanner:
    """
    Converts a ZombieCluster into an ordered list of file mutation steps.
    Each step is independently revertable.
    """

    def __init__(self, repo_path: Optional[Path] = None):
        self.repo_path = repo_path or Path(".")

    def plan(self, cluster: ZombieCluster) -> list[dict]:
        """
        Returns ordered list of file mutations.
        Each step dict contains: action, file, metadata, revert_cmd
        """
        steps: list[dict] = []

        # Group symbols by file
        by_file: dict[str, list[DeadSymbol]] = defaultdict(list)
        for sym in cluster.symbols:
            by_file[sym.file_path].append(sym)

        # Classify: full-file deletions vs. partial edits
        full_deletes, partial_deletes = self._classify_files(by_file)

        # Step 1: Delete entirely dead files
        for file_path in sorted(full_deletes):
            steps.append({
                "action": "delete_file",
                "file": file_path,
                "reason": "entire file is dead (all symbols confirmed dead)",
                "symbols": [s.fqn for s in by_file[file_path]],
                "revert_cmd": f"git checkout HEAD -- {file_path}",
            })

        # Step 2: Remove dead symbols from live files
        # Sort by line number DESCENDING so line numbers stay valid during edits
        for file_path in sorted(partial_deletes.keys()):
            symbols = partial_deletes[file_path]
            sorted_syms = sorted(symbols, key=lambda s: s.start_line, reverse=True)

            for sym in sorted_syms:
                steps.append({
                    "action": "remove_lines",
                    "file": file_path,
                    "start_line": sym.start_line,
                    "end_line": sym.end_line,
                    "symbol": sym.fqn,
                    "confidence": sym.confidence,
                    "revert_cmd": f"git diff HEAD -- {file_path} | git apply -R",
                })

        # Step 3: Clean up orphaned imports in partially-edited files
        for file_path in sorted(partial_deletes.keys()):
            steps.append({
                "action": "cleanup_imports",
                "file": file_path,
                "reason": "remove imports that only served deleted symbols",
            })

        return steps

    def _classify_files(
        self, by_file: dict[str, list[DeadSymbol]]
    ) -> tuple[list[str], dict[str, list[DeadSymbol]]]:
        """
        Separate files where everything is dead vs. files with mixed live/dead code.
        A file is fully dead if ALL top-level symbols in it are in the cluster.
        """
        full_deletes: list[str] = []
        partial_deletes: dict[str, list[DeadSymbol]] = {}

        for file_path, symbols in by_file.items():
            total_in_file = self._count_top_level_symbols(file_path)
            if total_in_file > 0 and len(symbols) >= total_in_file:
                full_deletes.append(file_path)
            else:
                partial_deletes[file_path] = symbols

        return full_deletes, partial_deletes

    def _count_top_level_symbols(self, file_path: str) -> int:
        """
        Count top-level function/class definitions in a file.
        Uses Python AST for .py files, falls back to a conservative count.
        """
        full_path = self.repo_path / file_path

        if not full_path.exists():
            logger.warning("File not found for symbol counting: %s", file_path)
            return 999  # Conservative — prevents accidental full-file delete

        if file_path.endswith(".py"):
            return self._count_python_symbols(full_path)
        elif file_path.endswith((".ts", ".tsx", ".js", ".jsx")):
            return self._count_ts_symbols(full_path)
        else:
            # Unknown language — be conservative
            return 999

    def _count_python_symbols(self, path: Path) -> int:
        """Count top-level classes and functions using Python's AST."""
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source)
            count = 0
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    count += 1
            return count
        except (SyntaxError, UnicodeDecodeError):
            logger.warning("Could not parse %s — assuming many symbols", path)
            return 999

    def _count_ts_symbols(self, path: Path) -> int:
        """
        Approximate top-level symbol count for TypeScript/JavaScript.
        Uses a simple regex-based heuristic — not a full parser.
        """
        import re

        try:
            source = path.read_text(encoding="utf-8")
            # Count: export function, export class, export const, function, class
            patterns = [
                r"^export\s+(default\s+)?(function|class|const|let|var)\b",
                r"^(function|class)\s+\w+",
            ]
            count = 0
            for line in source.splitlines():
                stripped = line.strip()
                for pattern in patterns:
                    if re.match(pattern, stripped):
                        count += 1
                        break
            return max(count, 1)  # At least 1 to avoid division issues
        except (UnicodeDecodeError, OSError):
            return 999

    def validate_plan(self, steps: list[dict], repo_path: Path) -> tuple[bool, list[str]]:
        """
        Validate that a deletion plan is safe to execute.
        Returns (valid, errors).
        """
        errors: list[str] = []

        for step in steps:
            file_path = repo_path / step["file"]

            if step["action"] == "delete_file":
                if not file_path.exists():
                    errors.append(f"Cannot delete {step['file']} — file does not exist")

            elif step["action"] == "remove_lines":
                if not file_path.exists():
                    errors.append(f"Cannot edit {step['file']} — file does not exist")
                else:
                    lines = file_path.read_text().splitlines()
                    if step["end_line"] > len(lines):
                        errors.append(
                            f"Line range {step['start_line']}-{step['end_line']} "
                            f"exceeds file length ({len(lines)}) in {step['file']}"
                        )

        return len(errors) == 0, errors
