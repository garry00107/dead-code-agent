"""
Graph Builder — constructs a directed call/import graph from source code.

Nodes are symbols (functions, classes, methods, constants).
Edges represent references (imports, function calls, attribute access).

The graph feeds into Tarjan's SCC algorithm to find zombie clusters.

Supported languages:
  - Python:     Full AST-based analysis (ast module)
  - TypeScript: Regex-based import/export and reference analysis

Design:
  - Conservative: if in doubt, ADD an edge (over-counting references
    keeps code alive — safe by design)
  - No LLM: 100% deterministic
  - Incremental-ready: could be extended to cache graphs between runs
"""

from __future__ import annotations

import ast
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Graph Data Structures
# ─────────────────────────────────────────────

@dataclass
class GraphNode:
    """A symbol in the call graph."""
    fqn: str                          # Fully-qualified name: "src.payments.process_order"
    file_path: str                    # Relative path from repo root
    start_line: int                   # 1-indexed
    end_line: int                     # 1-indexed, inclusive
    kind: str                         # "function", "class", "method", "constant"
    name: str                         # Simple name: "process_order"


@dataclass
class CodeGraph:
    """Directed graph of symbol references."""
    nodes: dict[str, GraphNode] = field(default_factory=dict)   # fqn → node
    edges: dict[str, set[str]] = field(default_factory=dict)    # fqn → set of FQNs it references
    file_symbols: dict[str, list[str]] = field(default_factory=dict)  # filepath → list of FQNs

    def add_node(self, node: GraphNode) -> None:
        self.nodes[node.fqn] = node
        self.edges.setdefault(node.fqn, set())
        self.file_symbols.setdefault(node.file_path, []).append(node.fqn)

    def add_edge(self, from_fqn: str, to_fqn: str) -> None:
        """Add directed edge: from_fqn references/calls to_fqn."""
        if from_fqn in self.nodes and to_fqn in self.nodes:
            self.edges.setdefault(from_fqn, set()).add(to_fqn)

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        return sum(len(targets) for targets in self.edges.values())


# ─────────────────────────────────────────────
# Graph Builder
# ─────────────────────────────────────────────

class GraphBuilder:
    """
    Builds a CodeGraph from a repository's source files.

    Phase 1: Walk all files, extract symbol nodes (functions, classes, etc.)
    Phase 2: Walk all files again, resolve references into edges.
    """

    EXCLUDE_DIRS = {
        ".venv", "venv", "node_modules", ".git", "__pycache__",
        ".tox", "dist", "build", ".next", ".mypy_cache", ".pytest_cache",
        "egg-info",
    }

    def __init__(self, repo_path: Path, target_path: Optional[str] = None):
        """
        Args:
            repo_path: Root of the repository.
            target_path: Optional subdirectory to scope analysis (e.g., "src/").
        """
        self.repo_path = repo_path
        self.target_path = repo_path / target_path if target_path else repo_path
        self.graph = CodeGraph()

        # Lookup tables built during Phase 1 for Phase 2 resolution
        self._simple_name_to_fqns: dict[str, list[str]] = {}  # "process_order" → ["src.pay.process_order"]
        self._module_to_fqns: dict[str, list[str]] = {}  # "src.payments" → ["src.payments.X", ...]

    def build(self) -> CodeGraph:
        """Build the full graph: nodes first, then edges."""
        logger.info("Building code graph from %s", self.target_path)

        # Phase 1: Extract all symbols as nodes
        self._extract_python_nodes()
        self._extract_typescript_nodes()

        logger.info("Phase 1 complete: %d nodes in %d files",
                     self.graph.node_count, len(self.graph.file_symbols))

        # Phase 2: Resolve references into edges
        self._resolve_python_edges()
        self._resolve_typescript_edges()

        logger.info("Phase 2 complete: %d edges", self.graph.edge_count)
        return self.graph

    # ─────────────────────────────────────────────
    # Phase 1: Node Extraction — Python
    # ─────────────────────────────────────────────

    def _extract_python_nodes(self) -> None:
        """
        Parse Python files and extract all symbol nodes.

        Handles:
          - Top-level functions (sync + async)
          - Classes with methods (including nested)
          - Module-level UPPER_CASE constants
          - Nested functions/closures (critical for decorator patterns)
        """
        for py_file in self._glob("**/*.py"):
            try:
                source = py_file.read_text(errors="replace")
                tree = ast.parse(source)
            except SyntaxError:
                logger.debug("Skipping unparseable Python file: %s", py_file)
                continue

            module = self._path_to_module(py_file)
            rel_path = str(py_file.relative_to(self.repo_path))

            self._extract_python_nodes_recursive(tree, module, rel_path, parent_fqn=module)

    def _extract_python_nodes_recursive(
        self, parent_node: ast.AST, module: str, rel_path: str, parent_fqn: str,
    ) -> None:
        """
        Recursively extract nodes from Python AST, handling:
          - Nested functions and closures
          - Class methods (including nested classes)
          - Module-level constants
        """
        for node in ast.iter_child_nodes(parent_node):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                fqn = f"{parent_fqn}.{node.name}" if parent_fqn != module else f"{module}.{node.name}"
                # Avoid double-adding if parent is module
                if parent_fqn != module:
                    fqn = f"{parent_fqn}.{node.name}"
                else:
                    fqn = f"{module}.{node.name}"
                self._add_python_node(fqn, rel_path, node, "function" if parent_fqn == module else "method")

                # Recurse into nested functions/closures
                self._extract_python_nodes_recursive(node, module, rel_path, parent_fqn=fqn)

            elif isinstance(node, ast.ClassDef):
                class_fqn = f"{parent_fqn}.{node.name}" if parent_fqn != module else f"{module}.{node.name}"
                if parent_fqn != module:
                    class_fqn = f"{parent_fqn}.{node.name}"
                else:
                    class_fqn = f"{module}.{node.name}"
                self._add_python_node(class_fqn, rel_path, node, "class")

                # Recurse into class body (methods, nested classes, etc.)
                self._extract_python_nodes_recursive(node, module, rel_path, parent_fqn=class_fqn)

            elif isinstance(node, ast.Assign) and parent_fqn == module:
                # Module-level constants only
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.isupper():
                        fqn = f"{module}.{target.id}"
                        self._add_python_node(
                            fqn, rel_path, node, "constant",
                            name_override=target.id,
                        )

    # ─────────────────────────────────────────────
    # Phase 1: Node Extraction — TypeScript
    # ─────────────────────────────────────────────

    def _extract_typescript_nodes(self) -> None:
        """
        Parse TypeScript/TSX/JSX files and extract symbol nodes.

        Uses regex (no ts-morph dependency) with patterns for:
          - Functions (sync + async, exported + internal)
          - Arrow function exports (const X = () => ...)
          - Classes (exported + abstract)
          - Class methods
          - Interfaces and type aliases
          - React component exports
        """
        # Regex patterns for TS symbol extraction
        func_re = re.compile(
            r"^(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*[(<]",
            re.MULTILINE,
        )
        class_re = re.compile(
            r"^(?:export\s+)?(?:abstract\s+)?class\s+(\w+)",
            re.MULTILINE,
        )
        const_re = re.compile(
            r"^(?:export\s+)?(?:const|let|var)\s+(\w+)\s*[=:]",
            re.MULTILINE,
        )
        interface_re = re.compile(
            r"^(?:export\s+)?(?:interface|type)\s+(\w+)",
            re.MULTILINE,
        )
        # Class method extraction
        method_re = re.compile(
            r"^\s+(?:(?:public|private|protected|static|async|abstract|readonly)\s+)?"
            r"(?:(?:get|set)\s+)?(\w+)\s*\(",
            re.MULTILINE,
        )
        # Default export: export default function/class X
        default_export_re = re.compile(
            r"^export\s+default\s+(?:function|class)\s+(\w+)",
            re.MULTILINE,
        )

        ts_extensions = ["**/*.ts", "**/*.tsx", "**/*.jsx"]

        for ext_pattern in ts_extensions:
            for ts_file in self._glob(ext_pattern):
                try:
                    content = ts_file.read_text(errors="replace")
                except OSError:
                    continue

                module = self._path_to_module(ts_file)
                rel_path = str(ts_file.relative_to(self.repo_path))
                lines = content.splitlines()

                # Track which class each method belongs to
                current_class: Optional[str] = None
                class_end_line: int = 0

                # Extract top-level symbols
                for pattern, kind in [
                    (func_re, "function"),
                    (class_re, "class"),
                    (const_re, "constant"),
                    (interface_re, "class"),  # treat interfaces as classes for graph purposes
                    (default_export_re, "function"),
                ]:
                    for match in pattern.finditer(content):
                        name = match.group(1)
                        fqn = f"{module}.{name}"
                        line_no = content[:match.start()].count("\n") + 1

                        # Estimate end line from next top-level construct or EOF
                        end_line = self._estimate_ts_end_line(content, match.start(), lines)

                        gn = GraphNode(
                            fqn=fqn,
                            file_path=rel_path,
                            start_line=line_no,
                            end_line=end_line,
                            kind=kind,
                            name=name,
                        )
                        self.graph.add_node(gn)
                        self._simple_name_to_fqns.setdefault(name, []).append(fqn)
                        self._module_to_fqns.setdefault(module, []).append(fqn)

                # Extract class methods
                for class_match in class_re.finditer(content):
                    class_name = class_match.group(1)
                    class_fqn = f"{module}.{class_name}"
                    class_start = class_match.start()
                    class_end = self._find_brace_end(content, class_start)

                    class_body = content[class_start:class_end]
                    class_start_line = content[:class_start].count("\n") + 1

                    for method_match in method_re.finditer(class_body):
                        method_name = method_match.group(1)
                        # Skip constructor, common keywords that aren't method names
                        if method_name in ("constructor", "if", "else", "return", "throw",
                                           "for", "while", "switch", "case", "class"):
                            continue
                        method_fqn = f"{class_fqn}.{method_name}"
                        method_line = class_start_line + class_body[:method_match.start()].count("\n")

                        gn = GraphNode(
                            fqn=method_fqn,
                            file_path=rel_path,
                            start_line=method_line,
                            end_line=method_line,  # approximation
                            kind="method",
                            name=method_name,
                        )
                        self.graph.add_node(gn)
                        self._simple_name_to_fqns.setdefault(method_name, []).append(method_fqn)
                        self._module_to_fqns.setdefault(module, []).append(method_fqn)

    def _estimate_ts_end_line(self, content: str, match_start: int, lines: list[str]) -> int:
        """Estimate the end line of a TS construct by finding the next top-level symbol."""
        line_no = content[:match_start].count("\n") + 1
        # Simple heuristic: scan forward for next non-indented definition
        top_level_re = re.compile(r"^(?:export\s+|async\s+)?(?:function|class|const|let|var|interface|type)\s", re.MULTILINE)
        remaining = content[match_start + 1:]
        next_match = top_level_re.search(remaining)
        if next_match:
            return line_no + remaining[:next_match.start()].count("\n")
        return len(lines)

    def _find_brace_end(self, content: str, start: int) -> int:
        """Find the matching closing brace for a class/function starting at `start`."""
        brace_pos = content.find("{", start)
        if brace_pos == -1:
            return min(start + 500, len(content))

        depth = 0
        i = brace_pos
        while i < len(content):
            if content[i] == "{":
                depth += 1
            elif content[i] == "}":
                depth -= 1
                if depth == 0:
                    return i + 1
            i += 1
        return len(content)

    def _add_python_node(
        self, fqn: str, rel_path: str, node: ast.AST,
        kind: str, name_override: Optional[str] = None,
    ) -> None:
        """Add a Python AST node to the graph."""
        name = name_override or getattr(node, "name", fqn.split(".")[-1])
        end_line = getattr(node, "end_lineno", getattr(node, "lineno", 0))
        gn = GraphNode(
            fqn=fqn,
            file_path=rel_path,
            start_line=getattr(node, "lineno", 0),
            end_line=end_line or 0,
            kind=kind,
            name=name,
        )
        self.graph.add_node(gn)
        self._simple_name_to_fqns.setdefault(name, []).append(fqn)
        module = ".".join(fqn.split(".")[:-1])
        self._module_to_fqns.setdefault(module, []).append(fqn)

    # ─────────────────────────────────────────────
    # Phase 2: Edge Resolution — Python
    # ─────────────────────────────────────────────

    def _resolve_python_edges(self) -> None:
        """
        Walk Python ASTs to find all references and create edges.

        Handles:
          - Import statements (absolute + relative)
          - Star imports (conservatively links to all symbols in target module)
          - Function calls and attribute access
          - Decorator references (function ← decorator edge)
          - Type annotations (function → type edge)
          - Base class inheritance (class → base edge)
          - Module-level calls (top-level register() patterns)
          - __init__.py re-exports
        """
        for py_file in self._glob("**/*.py"):
            try:
                source = py_file.read_text(errors="replace")
                tree = ast.parse(source)
            except SyntaxError:
                continue

            module = self._path_to_module(py_file)
            rel_path = str(py_file.relative_to(self.repo_path))

            # Collect import mappings for this file
            imports = self._collect_python_imports(tree, module)

            # Handle __init__.py re-exports: treat them as roots for their module
            if py_file.name == "__init__.py":
                self._handle_init_reexports(tree, module, imports)

            # Walk all nodes — including top-level statements
            self._resolve_python_file_edges(tree, module, imports)

    def _resolve_python_file_edges(
        self, tree: ast.Module, module: str, imports: dict[str, str],
    ) -> None:
        """Resolve edges for an entire file, including top-level statements."""
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                from_fqn = f"{module}.{node.name}"
                # Decorator edges: function depends on decorator
                self._resolve_decorator_edges(from_fqn, node.decorator_list, imports, module)
                # Type annotation edges
                self._resolve_annotation_edges(from_fqn, node, imports, module)
                # Body references
                self._resolve_references_in_body(from_fqn, node.body, imports, module)

            elif isinstance(node, ast.ClassDef):
                class_fqn = f"{module}.{node.name}"
                # Base class references
                for base in node.bases:
                    self._resolve_name_reference(class_fqn, base, imports, module)
                # Decorator edges (e.g., @dataclass)
                self._resolve_decorator_edges(class_fqn, node.decorator_list, imports, module)

                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        method_fqn = f"{class_fqn}.{item.name}"
                        self._resolve_decorator_edges(method_fqn, item.decorator_list, imports, module)
                        self._resolve_annotation_edges(method_fqn, item, imports, module)
                        self._resolve_references_in_body(method_fqn, item.body, imports, module)

            elif isinstance(node, ast.Expr):
                # Module-level calls: e.g., register(handler), app.include_router(router)
                self._resolve_module_level_call(node, module, imports)

    def _resolve_decorator_edges(
        self, from_fqn: str, decorators: list[ast.expr], imports: dict[str, str], module: str,
    ) -> None:
        """
        Create edges from a decorated symbol to its decorators.

        Handles:
          - Simple: @decorator → edge to decorator
          - Called: @decorator(args) → edge to decorator
          - Attribute: @module.decorator → edge via attribute resolution
        """
        for dec in decorators:
            if isinstance(dec, ast.Call):
                # @decorator(args) — resolve the callable
                self._resolve_name_reference(from_fqn, dec.func, imports, module)
                # Also resolve arguments (e.g., @register("name", handler=X))
                for arg in dec.args:
                    self._resolve_name_reference(from_fqn, arg, imports, module)
                for keyword in dec.keywords:
                    self._resolve_name_reference(from_fqn, keyword.value, imports, module)
            else:
                # @decorator — resolve directly
                self._resolve_name_reference(from_fqn, dec, imports, module)

    def _resolve_annotation_edges(
        self, from_fqn: str, func_node: ast.AST, imports: dict[str, str], module: str,
    ) -> None:
        """
        Create edges from a function to types referenced in its annotations.

        Covers: parameter types, return types, and variable annotations in the body.
        """
        if not isinstance(func_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return

        # Return annotation
        if func_node.returns:
            self._resolve_annotation_expr(from_fqn, func_node.returns, imports, module)

        # Parameter annotations
        for arg in func_node.args.args + func_node.args.posonlyargs + func_node.args.kwonlyargs:
            if arg.annotation:
                self._resolve_annotation_expr(from_fqn, arg.annotation, imports, module)
        if func_node.args.vararg and func_node.args.vararg.annotation:
            self._resolve_annotation_expr(from_fqn, func_node.args.vararg.annotation, imports, module)
        if func_node.args.kwarg and func_node.args.kwarg.annotation:
            self._resolve_annotation_expr(from_fqn, func_node.args.kwarg.annotation, imports, module)

    def _resolve_annotation_expr(
        self, from_fqn: str, annotation: ast.expr, imports: dict[str, str], module: str,
    ) -> None:
        """Resolve a type annotation expression to edges."""
        if isinstance(annotation, ast.Name):
            self._try_add_edge(from_fqn, annotation.id, imports, module)
        elif isinstance(annotation, ast.Attribute):
            self._resolve_name_reference(from_fqn, annotation, imports, module)
        elif isinstance(annotation, ast.Subscript):
            # Handle generics: List[X], Optional[Y], Dict[K, V]
            self._resolve_annotation_expr(from_fqn, annotation.value, imports, module)
            if isinstance(annotation.slice, ast.Tuple):
                for elt in annotation.slice.elts:
                    self._resolve_annotation_expr(from_fqn, elt, imports, module)
            else:
                self._resolve_annotation_expr(from_fqn, annotation.slice, imports, module)
        elif isinstance(annotation, ast.BinOp):
            # Handle X | Y union syntax (Python 3.10+)
            self._resolve_annotation_expr(from_fqn, annotation.left, imports, module)
            self._resolve_annotation_expr(from_fqn, annotation.right, imports, module)
        elif isinstance(annotation, ast.Constant):
            # String annotations: "ForwardRef"
            if isinstance(annotation.value, str):
                self._try_add_edge(from_fqn, annotation.value, imports, module)

    def _resolve_module_level_call(
        self, node: ast.Expr, module: str, imports: dict[str, str],
    ) -> None:
        """
        Handle module-level function calls, like:
            register(my_handler)
            app.include_router(router)

        These create edges from ALL symbols in the module to the arguments,
        because we don't know which symbol the registration affects.
        Conservative: better to over-connect.
        """
        if not isinstance(node.value, ast.Call):
            return

        call = node.value
        file_fqns = self.graph.file_symbols.get(
            next(
                (fp for fp, syms in self.graph.file_symbols.items()
                 if any(s.startswith(module + ".") for s in syms)),
                "",
            ), [],
        )

        # For each argument, try to resolve it as a symbol reference
        for arg in call.args:
            if isinstance(arg, ast.Name):
                for from_fqn in file_fqns:
                    self._try_add_edge(from_fqn, arg.id, imports, module)

    def _handle_init_reexports(
        self, tree: ast.Module, module: str, imports: dict[str, str],
    ) -> None:
        """
        Handle __init__.py re-exports.

        When __init__.py does `from .submodule import X`, symbol X should be
        reachable via the parent module. Create edges so that references to
        `parent.X` resolve to `parent.submodule.X`.
        """
        # The module for pkg/__init__.py is "pkg.__init__"
        # Strip ".__init__" to get the package name for proxy FQNs
        package_name = module.removesuffix(".__init__") if module.endswith(".__init__") else module

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level > 0:
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    local_name = alias.asname or alias.name
                    # If the import maps to a known FQN, create a proxy node or edge
                    if local_name in imports:
                        source_fqn = imports[local_name]
                        proxy_fqn = f"{package_name}.{local_name}"
                        # If source exists in graph, add edge from proxy to source
                        if source_fqn in self.graph.nodes and proxy_fqn not in self.graph.nodes:
                            # Create a proxy node in __init__.py
                            source_node = self.graph.nodes[source_fqn]
                            proxy = GraphNode(
                                fqn=proxy_fqn,
                                file_path=source_node.file_path,
                                start_line=source_node.start_line,
                                end_line=source_node.end_line,
                                kind=source_node.kind,
                                name=local_name,
                            )
                            self.graph.add_node(proxy)
                            self._simple_name_to_fqns.setdefault(local_name, []).append(proxy_fqn)
                            # Proxy references the real symbol
                            self.graph.add_edge(proxy_fqn, source_fqn)

    def _resolve_references_in_body(
        self, from_fqn: str, body: list, imports: dict[str, str], module: str,
    ) -> None:
        """Walk an AST body and add edges for all name references."""
        for node in ast.walk(ast.Module(body=body, type_ignores=[])):
            if isinstance(node, ast.Call):
                self._resolve_name_reference(from_fqn, node.func, imports, module)
            elif isinstance(node, ast.Name):
                self._try_add_edge(from_fqn, node.id, imports, module)
            elif isinstance(node, ast.Attribute):
                self._resolve_name_reference(from_fqn, node, imports, module)

    def _resolve_name_reference(
        self, from_fqn: str, node: ast.AST, imports: dict[str, str], module: str,
    ) -> None:
        """Resolve a Name or Attribute AST node to a graph edge."""
        if isinstance(node, ast.Name):
            self._try_add_edge(from_fqn, node.id, imports, module)
        elif isinstance(node, ast.Attribute):
            # Handle: obj.method — try both the full chain and just the attr
            self._resolve_name_reference(from_fqn, node.value, imports, module)
            if isinstance(node.value, ast.Name):
                # Try: imported_module.symbol
                qualified = f"{node.value.id}.{node.attr}"
                self._try_add_edge(from_fqn, qualified, imports, module)
            self._try_add_edge(from_fqn, node.attr, imports, module)

    def _try_add_edge(
        self, from_fqn: str, name: str, imports: dict[str, str], module: str,
    ) -> None:
        """Attempt to resolve a name to a known node and add an edge."""
        # Skip self-references (by simple name, not by FQN — avoids masking)
        from_simple = from_fqn.split(".")[-1]
        if name == from_simple and f"{module}.{name}" == from_fqn:
            return

        candidates = []

        # 1. Check imports: "from x import y" → y maps to x.y
        if name in imports:
            resolved = imports[name]
            if resolved in self.graph.nodes:
                candidates.append(resolved)
            # Also check sub-symbol: imported_module.symbol
            if "." in name:
                parts = name.split(".")
                if parts[0] in imports:
                    resolved = imports[parts[0]] + "." + ".".join(parts[1:])
                    if resolved in self.graph.nodes:
                        candidates.append(resolved)

        # 2. Same-module reference
        same_module = f"{module}.{name}"
        if same_module in self.graph.nodes:
            candidates.append(same_module)

        # 3. Simple name match (conservative — adds all matches)
        if name in self._simple_name_to_fqns:
            candidates.extend(self._simple_name_to_fqns[name])

        # Deduplicate and add edges
        for target in set(candidates):
            if target != from_fqn:
                self.graph.add_edge(from_fqn, target)

    def _collect_python_imports(self, tree: ast.Module, current_module: str) -> dict[str, str]:
        """
        Build mapping of local names → FQNs from import statements.

        Handles:
          - Absolute imports: import X, from X import Y
          - Relative imports: from .sub import Y, from ..parent import Z
          - Star imports: from X import * → maps ALL symbols in X
          - Aliased imports: from X import Y as Z
        """
        imports = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    local_name = alias.asname or alias.name
                    imports[local_name] = alias.name
            elif isinstance(node, ast.ImportFrom):
                if node.module is not None or node.level > 0:
                    # Resolve relative imports
                    if node.level > 0:
                        parts = current_module.split(".")
                        base = ".".join(parts[:max(1, len(parts) - node.level)])
                        full_module = f"{base}.{node.module}" if node.module else base
                    else:
                        full_module = node.module

                    for alias in node.names:
                        if alias.name == "*":
                            # Star import: conservatively map all known symbols from target module
                            if full_module in self._module_to_fqns:
                                for fqn in self._module_to_fqns[full_module]:
                                    simple = fqn.split(".")[-1]
                                    imports[simple] = fqn
                            continue

                        local_name = alias.asname or alias.name
                        imports[local_name] = f"{full_module}.{alias.name}"
        return imports

    # ─────────────────────────────────────────────
    # Phase 2: Edge Resolution — TypeScript
    # ─────────────────────────────────────────────

    def _resolve_typescript_edges(self) -> None:
        """
        Resolve TypeScript import/reference edges.

        Handles:
          - Named imports: import { X } from './y'
          - Default imports: import X from './y'
          - Barrel re-exports: export { X } from './y'
          - Per-symbol scope (not file-wide matching)
          - Dynamic imports: import('./module')
        """
        import_re = re.compile(
            r"import\s+(?:\{([^}]+)\}|(\w+))\s+from\s+['\"]([^'\"]+)['\"]",
        )
        # Barrel re-exports: export { X, Y as Z } from './module'
        reexport_re = re.compile(
            r"export\s+\{([^}]+)\}\s+from\s+['\"]([^'\"]+)['\"]",
        )
        # Dynamic import: import('./module')
        dynamic_import_re = re.compile(
            r"import\s*\(\s*['\"]([^'\"]+)['\"]\s*\)",
        )

        ts_extensions = ["**/*.ts", "**/*.tsx", "**/*.jsx"]

        for ext_pattern in ts_extensions:
            for ts_file in self._glob(ext_pattern):
                try:
                    content = ts_file.read_text(errors="replace")
                except OSError:
                    continue

                module = self._path_to_module(ts_file)
                rel_path = str(ts_file.relative_to(self.repo_path))

                # Collect imports for this file
                imports = self._collect_ts_imports(content, ts_file, import_re)

                # Handle barrel re-exports
                self._handle_ts_reexports(content, ts_file, module, reexport_re)

                # Per-symbol edge resolution
                file_fqns = self.graph.file_symbols.get(rel_path, [])
                for from_fqn in file_fqns:
                    node = self.graph.nodes[from_fqn]
                    # Get the source range for this symbol
                    lines = content.splitlines()
                    start = max(0, node.start_line - 1)
                    end = min(len(lines), node.end_line)
                    symbol_source = "\n".join(lines[start:end])

                    # Resolve references within this symbol's scope
                    for imported_name, resolved_fqn in imports.items():
                        if re.search(rf"\b{re.escape(imported_name)}\b", symbol_source):
                            if resolved_fqn in self.graph.nodes:
                                if resolved_fqn != from_fqn:
                                    self.graph.add_edge(from_fqn, resolved_fqn)
                            # Also try simple name match
                            elif imported_name in self._simple_name_to_fqns:
                                for target_fqn in self._simple_name_to_fqns[imported_name]:
                                    if target_fqn != from_fqn:
                                        self.graph.add_edge(from_fqn, target_fqn)

    def _collect_ts_imports(
        self, content: str, ts_file: Path, import_re: re.Pattern,
    ) -> dict[str, str]:
        """Build import mapping for a TypeScript file."""
        imports: dict[str, str] = {}
        for match in import_re.finditer(content):
            named = match.group(1)
            default = match.group(2)
            from_module = match.group(3)

            resolved_module = self._resolve_ts_module_path(from_module, ts_file)

            if named:
                for name_part in named.split(","):
                    name_part = name_part.strip()
                    if " as " in name_part:
                        original, alias = name_part.split(" as ", 1)
                        imports[alias.strip()] = f"{resolved_module}.{original.strip()}"
                    elif name_part:
                        imports[name_part] = f"{resolved_module}.{name_part}"
            if default:
                # Default import → try matching the module's default export or same-named symbol
                imports[default] = f"{resolved_module}.{default}"

        return imports

    def _handle_ts_reexports(
        self, content: str, ts_file: Path, module: str, reexport_re: re.Pattern,
    ) -> None:
        """Handle barrel re-exports: export { X } from './submodule'."""
        for match in reexport_re.finditer(content):
            names = match.group(1)
            from_module = match.group(2)
            resolved_module = self._resolve_ts_module_path(from_module, ts_file)

            for name_part in names.split(","):
                name_part = name_part.strip()
                if " as " in name_part:
                    original, alias = name_part.split(" as ", 1)
                    original = original.strip()
                    alias = alias.strip()
                else:
                    original = name_part
                    alias = name_part

                source_fqn = f"{resolved_module}.{original}"
                proxy_fqn = f"{module}.{alias}"

                # Create proxy node if source exists
                if source_fqn in self.graph.nodes and proxy_fqn not in self.graph.nodes:
                    source_node = self.graph.nodes[source_fqn]
                    proxy = GraphNode(
                        fqn=proxy_fqn,
                        file_path=str(ts_file.relative_to(self.repo_path)),
                        start_line=1,
                        end_line=1,
                        kind=source_node.kind,
                        name=alias,
                    )
                    self.graph.add_node(proxy)
                    self._simple_name_to_fqns.setdefault(alias, []).append(proxy_fqn)
                    self.graph.add_edge(proxy_fqn, source_fqn)

    def _resolve_ts_module_path(self, from_module: str, ts_file: Path) -> str:
        """Resolve a TypeScript import path to a dotted module name."""
        if from_module.startswith("."):
            ts_dir = ts_file.parent
            resolved = (ts_dir / from_module).resolve()
            try:
                if resolved.is_relative_to(self.repo_path.resolve()):
                    return self._path_to_module(
                        resolved.relative_to(self.repo_path.resolve())
                    )
            except (ValueError, TypeError):
                pass
        return from_module

    # ─────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────

    def _glob(self, pattern: str) -> list[Path]:
        """Glob for files, excluding common non-source directories."""
        results = []
        for p in self.target_path.glob(pattern):
            if any(part in self.EXCLUDE_DIRS for part in p.parts):
                continue
            results.append(p)
        return results

    def _path_to_module(self, file_path: Path) -> str:
        """Convert file path to dotted module name."""
        try:
            rel = file_path.relative_to(self.repo_path)
        except ValueError:
            rel = file_path
        module = str(rel).replace(os.sep, ".").replace("/", ".")
        for ext in (".py", ".ts", ".tsx", ".js", ".jsx"):
            if module.endswith(ext):
                module = module[: -len(ext)]
                break
        return module
