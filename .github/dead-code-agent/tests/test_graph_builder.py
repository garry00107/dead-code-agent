"""
Tests for GraphBuilder — verifies AST-based graph construction.

Coverage areas:
  - Python: function/class/method/constant extraction, nested closures,
    decorator edges, type annotation edges, __init__.py re-exports,
    star imports, module-level calls
  - TypeScript: function/class/method extraction, barrel re-exports,
    per-symbol scope resolution, .jsx/.tsx support
  - Edge resolution: import-based, same-module, cross-module
  - Error handling: syntax errors, excluded dirs, empty repos
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graph_builder import GraphBuilder, CodeGraph


# ─── Helpers ────────────────────────────────────────────────────────

def _create_repo(tmp_path: Path, files: dict[str, str]) -> Path:
    for relpath, content in files.items():
        fp = tmp_path / relpath
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
    return tmp_path


# ─── Python Node Extraction ───────────────────────────────────────

class TestPythonNodeExtraction:
    def test_extracts_functions(self, tmp_path):
        repo = _create_repo(tmp_path, {
            "src/utils.py": "def helper():\n    pass\n\ndef compute():\n    pass\n",
        })
        builder = GraphBuilder(repo)
        graph = builder.build()
        fqns = set(graph.nodes.keys())
        assert "src.utils.helper" in fqns
        assert "src.utils.compute" in fqns

    def test_extracts_classes_and_methods(self, tmp_path):
        repo = _create_repo(tmp_path, {
            "src/models.py": (
                "class Order:\n"
                "    def process(self):\n"
                "        pass\n"
                "    def cancel(self):\n"
                "        pass\n"
            ),
        })
        builder = GraphBuilder(repo)
        graph = builder.build()
        fqns = set(graph.nodes.keys())
        assert "src.models.Order" in fqns
        assert "src.models.Order.process" in fqns
        assert "src.models.Order.cancel" in fqns

    def test_extracts_constants(self, tmp_path):
        repo = _create_repo(tmp_path, {
            "config.py": "MAX_RETRIES = 3\nAPI_URL = 'https://example.com'\n",
        })
        builder = GraphBuilder(repo)
        graph = builder.build()
        fqns = set(graph.nodes.keys())
        assert "config.MAX_RETRIES" in fqns
        assert "config.API_URL" in fqns

    def test_extracts_nested_functions(self, tmp_path):
        """Nested functions/closures must be captured — common in decorator patterns."""
        repo = _create_repo(tmp_path, {
            "decorators.py": (
                "def outer():\n"
                "    def inner():\n"
                "        pass\n"
                "    return inner\n"
            ),
        })
        builder = GraphBuilder(repo)
        graph = builder.build()
        fqns = set(graph.nodes.keys())
        assert "decorators.outer" in fqns
        assert "decorators.outer.inner" in fqns

    def test_extracts_async_functions(self, tmp_path):
        repo = _create_repo(tmp_path, {
            "async_mod.py": "async def fetch():\n    pass\n",
        })
        builder = GraphBuilder(repo)
        graph = builder.build()
        assert "async_mod.fetch" in graph.nodes

    def test_skips_excluded_dirs(self, tmp_path):
        repo = _create_repo(tmp_path, {
            "src/app.py": "def real():\n    pass\n",
            "node_modules/dep/mod.py": "def fake():\n    pass\n",
            ".venv/lib/site.py": "def venv_func():\n    pass\n",
        })
        builder = GraphBuilder(repo)
        graph = builder.build()
        fqns = set(graph.nodes.keys())
        assert "src.app.real" in fqns
        assert not any("fake" in f for f in fqns)
        assert not any("venv_func" in f for f in fqns)


# ─── TypeScript Node Extraction ──────────────────────────────────

class TestTypeScriptNodeExtraction:
    def test_extracts_ts_functions(self, tmp_path):
        repo = _create_repo(tmp_path, {
            "src/utils.ts": "export function helper() { return 1; }\nfunction internal() {}\n",
        })
        builder = GraphBuilder(repo)
        graph = builder.build()
        fqns = set(graph.nodes.keys())
        assert "src.utils.helper" in fqns
        assert "src.utils.internal" in fqns

    def test_extracts_ts_classes(self, tmp_path):
        repo = _create_repo(tmp_path, {
            "src/service.ts": "export class OrderService {\n}\n",
        })
        builder = GraphBuilder(repo)
        graph = builder.build()
        assert "src.service.OrderService" in graph.nodes

    def test_extracts_ts_class_methods(self, tmp_path):
        """TS class methods must be extracted as separate nodes."""
        repo = _create_repo(tmp_path, {
            "src/service.ts": (
                "export class OrderService {\n"
                "    async processOrder(id: string) {\n"
                "        return id;\n"
                "    }\n"
                "    private validateOrder() {\n"
                "        return true;\n"
                "    }\n"
                "}\n"
            ),
        })
        builder = GraphBuilder(repo)
        graph = builder.build()
        fqns = set(graph.nodes.keys())
        assert "src.service.OrderService" in fqns
        assert "src.service.OrderService.processOrder" in fqns
        assert "src.service.OrderService.validateOrder" in fqns

    def test_extracts_tsx_components(self, tmp_path):
        """TSX/JSX files must be scanned."""
        repo = _create_repo(tmp_path, {
            "src/Button.tsx": "export function Button() { return <button />; }\n",
        })
        builder = GraphBuilder(repo)
        graph = builder.build()
        assert "src.Button.Button" in graph.nodes

    def test_extracts_jsx_files(self, tmp_path):
        repo = _create_repo(tmp_path, {
            "src/App.jsx": "export function App() { return <div />; }\n",
        })
        builder = GraphBuilder(repo)
        graph = builder.build()
        assert "src.App.App" in graph.nodes

    def test_extracts_arrow_function_exports(self, tmp_path):
        repo = _create_repo(tmp_path, {
            "src/helpers.ts": "export const formatDate = (d: Date) => d.toString();\n",
        })
        builder = GraphBuilder(repo)
        graph = builder.build()
        assert "src.helpers.formatDate" in graph.nodes


# ─── Python Edge Resolution ──────────────────────────────────────

class TestPythonEdgeResolution:
    def test_function_call_creates_edge(self, tmp_path):
        repo = _create_repo(tmp_path, {
            "src/a.py": "from src.b import helper\n\ndef main():\n    helper()\n",
            "src/b.py": "def helper():\n    pass\n",
        })
        builder = GraphBuilder(repo)
        graph = builder.build()
        main_edges = graph.edges.get("src.a.main", set())
        assert "src.b.helper" in main_edges

    def test_same_module_reference(self, tmp_path):
        repo = _create_repo(tmp_path, {
            "module.py": "def a():\n    b()\n\ndef b():\n    pass\n",
        })
        builder = GraphBuilder(repo)
        graph = builder.build()
        a_edges = graph.edges.get("module.a", set())
        assert "module.b" in a_edges

    def test_class_inheritance_edge(self, tmp_path):
        repo = _create_repo(tmp_path, {
            "base.py": "class Base:\n    pass\n",
            "child.py": "from base import Base\n\nclass Child(Base):\n    pass\n",
        })
        builder = GraphBuilder(repo)
        graph = builder.build()
        child_edges = graph.edges.get("child.Child", set())
        assert "base.Base" in child_edges

    def test_no_self_reference(self, tmp_path):
        repo = _create_repo(tmp_path, {
            "module.py": "def recursive():\n    recursive()\n",
        })
        builder = GraphBuilder(repo)
        graph = builder.build()
        edges = graph.edges.get("module.recursive", set())
        assert "module.recursive" not in edges

    def test_decorator_creates_edge(self, tmp_path):
        """Decorated functions must have edges TO the decorator."""
        repo = _create_repo(tmp_path, {
            "decs.py": "def my_decorator(fn):\n    return fn\n",
            "handlers.py": (
                "from decs import my_decorator\n\n"
                "@my_decorator\n"
                "def handler():\n"
                "    pass\n"
            ),
        })
        builder = GraphBuilder(repo)
        graph = builder.build()
        handler_edges = graph.edges.get("handlers.handler", set())
        assert "decs.my_decorator" in handler_edges

    def test_decorator_with_arguments_creates_edge(self, tmp_path):
        """@decorator(arg) should create edges to both decorator and arg."""
        repo = _create_repo(tmp_path, {
            "decs.py": "def register(name):\n    def wrapper(fn):\n        return fn\n    return wrapper\n",
            "handlers.py": (
                "from decs import register\n\n"
                "@register('my_handler')\n"
                "def handler():\n"
                "    pass\n"
            ),
        })
        builder = GraphBuilder(repo)
        graph = builder.build()
        handler_edges = graph.edges.get("handlers.handler", set())
        assert "decs.register" in handler_edges

    def test_type_annotation_creates_edge(self, tmp_path):
        """Type annotations (param types, return types) must create edges."""
        repo = _create_repo(tmp_path, {
            "types.py": "class Order:\n    pass\n\nclass Result:\n    pass\n",
            "service.py": (
                "from types import Order, Result\n\n"
                "def process(order: Order) -> Result:\n"
                "    pass\n"
            ),
        })
        builder = GraphBuilder(repo)
        graph = builder.build()
        process_edges = graph.edges.get("service.process", set())
        assert "types.Order" in process_edges
        assert "types.Result" in process_edges

    def test_star_import_resolves_conservatively(self, tmp_path):
        """from module import * must conservatively include all symbols."""
        repo = _create_repo(tmp_path, {
            "utils.py": "def helper_a():\n    pass\n\ndef helper_b():\n    pass\n",
            "main.py": (
                "from utils import *\n\n"
                "def run():\n"
                "    helper_a()\n"
            ),
        })
        builder = GraphBuilder(repo)
        graph = builder.build()
        run_edges = graph.edges.get("main.run", set())
        assert "utils.helper_a" in run_edges

    def test_init_reexport_creates_proxy(self, tmp_path):
        """__init__.py re-exports should create proxy nodes."""
        repo = _create_repo(tmp_path, {
            "pkg/__init__.py": "from .core import Engine\n",
            "pkg/core.py": "class Engine:\n    pass\n",
            "app.py": "from pkg import Engine\n\ndef start():\n    Engine()\n",
        })
        builder = GraphBuilder(repo)
        graph = builder.build()
        # Proxy should exist
        assert "pkg.Engine" in graph.nodes
        # Proxy should reference the real symbol
        proxy_edges = graph.edges.get("pkg.Engine", set())
        assert "pkg.core.Engine" in proxy_edges


# ─── TypeScript Edge Resolution ──────────────────────────────────

class TestTypeScriptEdgeResolution:
    def test_named_import_creates_edge(self, tmp_path):
        repo = _create_repo(tmp_path, {
            "src/utils.ts": "export function helper() { return 1; }\n",
            "src/main.ts": (
                "import { helper } from './utils';\n\n"
                "export function run() {\n"
                "    return helper();\n"
                "}\n"
            ),
        })
        builder = GraphBuilder(repo)
        graph = builder.build()
        run_edges = graph.edges.get("src.main.run", set())
        assert "src.utils.helper" in run_edges

    def test_barrel_reexport_creates_proxy(self, tmp_path):
        """export { X } from './y' should create a proxy node."""
        repo = _create_repo(tmp_path, {
            "src/core.ts": "export function Engine() {}\n",
            "src/index.ts": "export { Engine } from './core';\n",
        })
        builder = GraphBuilder(repo)
        graph = builder.build()
        # Proxy should exist in the barrel module
        assert "src.index.Engine" in graph.nodes
        proxy_edges = graph.edges.get("src.index.Engine", set())
        assert "src.core.Engine" in proxy_edges


# ─── Graph Properties ─────────────────────────────────────────────

class TestGraphProperties:
    def test_empty_repo(self, tmp_path):
        builder = GraphBuilder(tmp_path)
        graph = builder.build()
        assert graph.node_count == 0
        assert graph.edge_count == 0

    def test_file_symbols_tracking(self, tmp_path):
        repo = _create_repo(tmp_path, {
            "module.py": "def a():\n    pass\n\ndef b():\n    pass\n",
        })
        builder = GraphBuilder(repo)
        graph = builder.build()
        assert "module.py" in graph.file_symbols
        assert len(graph.file_symbols["module.py"]) == 2

    def test_syntax_errors_skipped(self, tmp_path):
        repo = _create_repo(tmp_path, {
            "good.py": "def valid():\n    pass\n",
            "bad.py": "def broken(\n",  # SyntaxError
        })
        builder = GraphBuilder(repo)
        graph = builder.build()
        assert "good.valid" in graph.nodes
        assert graph.node_count == 1  # bad.py skipped

    def test_target_path_scoping(self, tmp_path):
        """When target_path is set, only files in that subtree are analysed."""
        repo = _create_repo(tmp_path, {
            "src/app.py": "def app_func():\n    pass\n",
            "tests/test_app.py": "def test_func():\n    pass\n",
        })
        builder = GraphBuilder(repo, target_path="src")
        graph = builder.build()
        fqns = set(graph.nodes.keys())
        assert "src.app.app_func" in fqns or "app.app_func" in fqns
        assert not any("test_func" in f for f in fqns)
