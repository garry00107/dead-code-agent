"""
Root Detector — orchestrates all 4 tiers of live entry point detection.

The root set defines the "alive" boundary. Everything reachable from a root
is alive; everything else is a candidate for deletion. Getting this wrong
in either direction has consequences:

  - Miss a root → live code gets deleted → production breaks (CATASTROPHIC)
  - Over-classify a root → dead code survives → tool is less useful (SAFE)

Design principle: aggressively over-classify. When in doubt, it's a root.

Tiers:
  1. Explicit:   main(), __all__, CLI entry points, test functions
  2. Framework:  Django views, FastAPI routes, NestJS controllers, etc.
  2.5 Learned:   Patterns from learned_patterns.yml (human-confirmed)
  3. Declared:   Manual entries in roots.yml (human escape valve)
"""

from __future__ import annotations

import ast
import fnmatch
import logging
import os
import re
import sys
from pathlib import Path
from typing import Optional

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from framework_detectors import (
    ALL_DETECTORS,
    FrameworkDetector,
    LearnedPatternDetector,
)

logger = logging.getLogger(__name__)


class RootDetector:
    """
    Orchestrates root detection across all tiers.

    Returns a set of fully-qualified symbol names (FQNs) that are
    considered "alive" entry points. The graph traversal uses this
    set to determine which SCCs are reachable from live code.
    """

    def __init__(
        self,
        repo_path: Path,
        roots_yml_path: Optional[Path] = None,
        learned_patterns_path: Optional[Path] = None,
    ):
        self.repo_path = repo_path
        self._roots_yml = roots_yml_path or (
            repo_path / ".github" / "dead-code-agent" / "roots.yml"
        )
        self._learned_patterns = learned_patterns_path or (
            repo_path / ".github" / "dead-code-agent" / "learned_patterns.yml"
        )

    def find_all_roots(self) -> dict[str, set[str]]:
        """
        Run all tiers and return roots grouped by source.

        Returns:
            {
                "tier1_explicit": {"module.main", ...},
                "tier2_framework": {"views.OrderView", ...},
                "tier2.5_learned": {"handlers.my_func", ...},
                "tier3_declared": {"src.rpc.*", ...},
            }
        """
        result = {}

        # ── Tier 1: Explicit roots ─────────────────────────────────
        tier1 = self._find_explicit_roots()
        result["tier1_explicit"] = tier1
        logger.info("Tier 1 (explicit): %d roots", len(tier1))

        # ── Tier 2: Framework roots ────────────────────────────────
        tier2 = self._find_framework_roots()
        result["tier2_framework"] = tier2
        logger.info("Tier 2 (framework): %d roots", len(tier2))

        # ── Tier 2.5: Learned patterns ─────────────────────────────
        tier25 = self._find_learned_roots()
        result["tier2.5_learned"] = tier25
        logger.info("Tier 2.5 (learned): %d roots", len(tier25))

        # ── Tier 3: Declared roots ─────────────────────────────────
        tier3 = self._find_declared_roots()
        result["tier3_declared"] = tier3
        logger.info("Tier 3 (declared): %d roots", len(tier3))

        total = sum(len(v) for v in result.values())
        logger.info("Total roots across all tiers: %d", total)
        return result

    def get_root_set(self) -> set[str]:
        """Return the flat union of all roots (for graph traversal)."""
        roots_by_tier = self.find_all_roots()
        return set().union(*roots_by_tier.values())

    def is_root(self, fqn: str, roots: Optional[set[str]] = None) -> bool:
        """
        Check if an FQN is a root, supporting glob patterns from Tier 3.
        """
        if roots is None:
            roots = self.get_root_set()

        # Exact match
        if fqn in roots:
            return True

        # Glob match (for Tier 3 patterns with * and **)
        for pattern in roots:
            if "*" in pattern:
                if fnmatch.fnmatch(fqn, pattern):
                    return True

        return False

    # ─────────────────────────────────────────────
    # Tier 1: Explicit Roots
    # ─────────────────────────────────────────────

    def _find_explicit_roots(self) -> set[str]:
        """
        Detect provably-live symbols with zero ambiguity:
          - if __name__ == "__main__" blocks
          - __all__ exports in __init__.py
          - CLI entry points from pyproject.toml/setup.py
          - test_* functions and Test* classes
          - TypeScript export default from index files
        """
        roots = set()
        roots |= self._find_python_main_blocks()
        roots |= self._find_python_all_exports()
        roots |= self._find_cli_entry_points()
        roots |= self._find_test_symbols()
        roots |= self._find_ts_index_exports()
        return roots

    def _find_python_main_blocks(self) -> set[str]:
        """Find files with if __name__ == '__main__': blocks."""
        roots = set()
        exclude = {".venv", "venv", "node_modules", ".git", "__pycache__"}
        for py_file in self.repo_path.rglob("*.py"):
            if any(part in exclude for part in py_file.parts):
                continue
            try:
                tree = ast.parse(py_file.read_text(errors="replace"))
            except SyntaxError:
                continue

            for node in ast.walk(tree):
                if isinstance(node, ast.If):
                    # Detect: if __name__ == "__main__"
                    if self._is_name_main_check(node):
                        fqn = self._path_to_module(py_file) + ".__main__"
                        roots.add(fqn)
                        # Also add all functions defined at module level
                        for item in ast.iter_child_nodes(tree):
                            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                roots.add(self._path_to_module(py_file) + "." + item.name)
                        break
        return roots

    def _find_python_all_exports(self) -> set[str]:
        """Find symbols listed in __all__ in __init__.py files."""
        roots = set()
        exclude = {".venv", "venv", "node_modules", ".git", "__pycache__"}
        for init_file in self.repo_path.rglob("__init__.py"):
            if any(part in exclude for part in init_file.parts):
                continue
            try:
                tree = ast.parse(init_file.read_text(errors="replace"))
            except SyntaxError:
                continue

            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id == "__all__":
                            if isinstance(node.value, (ast.List, ast.Tuple)):
                                for elt in node.value.elts:
                                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                        module = self._path_to_module(init_file)
                                        # Remove .__init__ suffix
                                        module = re.sub(r"\.__init__$", "", module)
                                        roots.add(f"{module}.{elt.value}")
        return roots

    def _find_cli_entry_points(self) -> set[str]:
        """Find CLI entry points from pyproject.toml [project.scripts]."""
        roots = set()
        pyproject = self.repo_path / "pyproject.toml"
        if pyproject.exists():
            try:
                content = pyproject.read_text()
                # Simple TOML parsing for [project.scripts] section
                in_scripts = False
                for line in content.splitlines():
                    line = line.strip()
                    if line == "[project.scripts]":
                        in_scripts = True
                        continue
                    if in_scripts:
                        if line.startswith("["):
                            break
                        if "=" in line:
                            _, _, value = line.partition("=")
                            value = value.strip().strip('"').strip("'")
                            # Format: "module.path:function"
                            if ":" in value:
                                module_part, func = value.rsplit(":", 1)
                                roots.add(f"{module_part}.{func}")
                            else:
                                roots.add(value)
            except (OSError, ValueError):
                pass

        # Also check setup.py/setup.cfg for console_scripts
        setup_py = self.repo_path / "setup.py"
        if setup_py.exists():
            try:
                content = setup_py.read_text(errors="replace")
                # Extract console_scripts entries
                for match in re.finditer(r"['\"](\w[\w.]+:\w+)['\"]", content):
                    entry = match.group(1)
                    module_part, func = entry.rsplit(":", 1)
                    roots.add(f"{module_part}.{func}")
            except OSError:
                pass

        return roots

    def _find_test_symbols(self) -> set[str]:
        """Find test functions and test classes — always alive."""
        roots = set()
        exclude = {".venv", "venv", "node_modules", ".git", "__pycache__"}

        for py_file in self.repo_path.rglob("*.py"):
            if any(part in exclude for part in py_file.parts):
                continue
            # Only scan test files
            if not (py_file.name.startswith("test_") or py_file.name.endswith("_test.py")):
                continue

            try:
                tree = ast.parse(py_file.read_text(errors="replace"))
            except SyntaxError:
                continue

            module = self._path_to_module(py_file)
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name.startswith("test_"):
                        roots.add(f"{module}.{node.name}")
                elif isinstance(node, ast.ClassDef):
                    if node.name.startswith("Test"):
                        roots.add(f"{module}.{node.name}")
                        for item in node.body:
                            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                roots.add(f"{module}.{node.name}.{item.name}")

        return roots

    def _find_ts_index_exports(self) -> set[str]:
        """Find default and named exports from TypeScript index files."""
        roots = set()
        exclude = {"node_modules", ".git", "dist", "build", ".next"}

        for ext in ["index.ts", "index.tsx", "index.js"]:
            for f in self.repo_path.rglob(ext):
                if any(part in exclude for part in f.parts):
                    continue
                try:
                    content = f.read_text(errors="replace")
                except OSError:
                    continue

                module = self._path_to_module(f)
                # export default
                if re.search(r"export\s+default\s+", content):
                    roots.add(f"{module}.default")
                # export { name1, name2 }
                for match in re.finditer(r"export\s*\{([^}]+)\}", content):
                    for name in match.group(1).split(","):
                        name = name.strip().split(" as ")[0].strip()
                        if name:
                            roots.add(f"{module}.{name}")

        return roots

    # ─────────────────────────────────────────────
    # Tier 2: Framework Roots
    # ─────────────────────────────────────────────

    def _find_framework_roots(self) -> set[str]:
        """Run all framework detectors and collect roots."""
        roots = set()
        for detector_cls in ALL_DETECTORS:
            detector = detector_cls()
            if detector.detect_framework(self.repo_path):
                logger.info("  Detected framework: %s", detector.name)
                framework_roots = detector.find_roots(self.repo_path)
                logger.info("  → %d roots from %s", len(framework_roots), detector.name)
                roots |= framework_roots
        return roots

    # ─────────────────────────────────────────────
    # Tier 2.5: Learned Patterns
    # ─────────────────────────────────────────────

    def _find_learned_roots(self) -> set[str]:
        """Apply confirmed learned patterns from YAML."""
        detector = LearnedPatternDetector(patterns_path=self._learned_patterns)
        if detector.detect_framework(self.repo_path):
            return detector.find_roots(self.repo_path)
        return set()

    # ─────────────────────────────────────────────
    # Tier 3: Declared Roots
    # ─────────────────────────────────────────────

    def _find_declared_roots(self) -> set[str]:
        """Load roots from roots.yml (human-maintained)."""
        if not self._roots_yml.exists():
            return set()

        roots = set()
        try:
            try:
                import yaml
                data = yaml.safe_load(self._roots_yml.read_text()) or {}
            except ImportError:
                # Minimal parsing without PyYAML
                data = self._parse_roots_yml_minimal()

            for entry in data.get("always_live", []) or []:
                if isinstance(entry, str) and entry.strip():
                    roots.add(entry.strip())

        except Exception as exc:
            logger.warning("Failed to load roots.yml: %s", exc)

        return roots

    def _parse_roots_yml_minimal(self) -> dict:
        """Parse roots.yml without PyYAML — just extract the always_live list."""
        entries = []
        in_always_live = False
        content = self._roots_yml.read_text()

        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("#") or not stripped:
                continue
            if stripped == "always_live:":
                in_always_live = True
                continue
            if in_always_live:
                if stripped.startswith("- "):
                    entry = stripped[2:].strip().strip('"').strip("'")
                    if entry and not entry.startswith("#"):
                        entries.append(entry)
                elif not stripped.startswith("-"):
                    # Hit a new top-level key
                    break

        return {"always_live": entries}

    # ─────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────

    @staticmethod
    def _is_name_main_check(node: ast.If) -> bool:
        """Check if an If node is: if __name__ == '__main__'"""
        test = node.test
        if isinstance(test, ast.Compare):
            if (
                len(test.ops) == 1
                and isinstance(test.ops[0], ast.Eq)
                and len(test.comparators) == 1
            ):
                left = test.left
                right = test.comparators[0]
                # Check both orderings
                if isinstance(left, ast.Name) and left.id == "__name__":
                    if isinstance(right, ast.Constant) and right.value == "__main__":
                        return True
                if isinstance(right, ast.Name) and right.id == "__name__":
                    if isinstance(left, ast.Constant) and left.value == "__main__":
                        return True
        return False

    def _path_to_module(self, file_path: Path) -> str:
        """Convert a file path to a dotted module name."""
        rel = file_path.relative_to(self.repo_path)
        module = str(rel).replace(os.sep, ".").replace("/", ".")
        for ext in (".py", ".ts", ".tsx", ".js", ".jsx"):
            if module.endswith(ext):
                module = module[: -len(ext)]
                break
        return module
