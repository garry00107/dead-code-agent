"""
Tests for RootDetector — verifies all 4 tiers of root detection.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from root_detector import RootDetector


# ─── Helpers ────────────────────────────────────────────────────────

def _create_repo(tmp_path: Path, files: dict[str, str]) -> Path:
    """Create a mock repo with the given files."""
    for relpath, content in files.items():
        fp = tmp_path / relpath
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content)
    return tmp_path


# ─── Tier 1: Explicit Roots ────────────────────────────────────────

class TestExplicitRoots:
    def test_detects_main_block(self, tmp_path):
        repo = _create_repo(tmp_path, {
            "app.py": 'def run():\n    pass\n\nif __name__ == "__main__":\n    run()\n',
        })
        detector = RootDetector(repo)
        roots = detector._find_explicit_roots()
        assert any("__main__" in r for r in roots)
        assert any("run" in r for r in roots)

    def test_detects_all_exports(self, tmp_path):
        repo = _create_repo(tmp_path, {
            "mypackage/__init__.py": '__all__ = ["Processor", "Handler"]\n',
        })
        detector = RootDetector(repo)
        roots = detector._find_explicit_roots()
        assert any("Processor" in r for r in roots)
        assert any("Handler" in r for r in roots)

    def test_detects_test_functions(self, tmp_path):
        repo = _create_repo(tmp_path, {
            "tests/test_core.py": (
                "def test_add():\n    assert 1 + 1 == 2\n\n"
                "class TestMath:\n    def test_sub(self):\n        pass\n"
            ),
        })
        detector = RootDetector(repo)
        roots = detector._find_explicit_roots()
        assert any("test_add" in r for r in roots)
        assert any("TestMath" in r for r in roots)
        assert any("test_sub" in r for r in roots)

    def test_detects_pyproject_scripts(self, tmp_path):
        repo = _create_repo(tmp_path, {
            "pyproject.toml": (
                '[project.scripts]\n'
                'my-cli = "mypackage.cli:main"\n'
            ),
        })
        detector = RootDetector(repo)
        roots = detector._find_explicit_roots()
        assert any("main" in r for r in roots)


# ─── Tier 2: Framework Roots ──────────────────────────────────────

class TestFrameworkRoots:
    def test_detects_django_views(self, tmp_path):
        repo = _create_repo(tmp_path, {
            "requirements.txt": "django==4.2\n",
            "views.py": (
                "from django.views import View\n\n"
                "class OrderView(View):\n"
                "    def get(self, request):\n"
                "        pass\n"
            ),
        })
        detector = RootDetector(repo)
        roots = detector._find_framework_roots()
        assert any("OrderView" in r for r in roots)

    def test_detects_fastapi_routes(self, tmp_path):
        repo = _create_repo(tmp_path, {
            "requirements.txt": "fastapi==0.100\n",
            "routes.py": (
                "from fastapi import FastAPI\n"
                "app = FastAPI()\n\n"
                "@app.get('/orders')\n"
                "async def list_orders():\n"
                "    pass\n"
            ),
        })
        detector = RootDetector(repo)
        roots = detector._find_framework_roots()
        assert any("list_orders" in r for r in roots)

    def test_detects_flask_routes(self, tmp_path):
        repo = _create_repo(tmp_path, {
            "requirements.txt": "flask==3.0\n",
            "app.py": (
                "from flask import Flask\n"
                "app = Flask(__name__)\n\n"
                "@app.route('/health')\n"
                "def health_check():\n"
                "    return 'ok'\n"
            ),
        })
        detector = RootDetector(repo)
        roots = detector._find_framework_roots()
        assert any("health_check" in r for r in roots)

    def test_detects_celery_tasks(self, tmp_path):
        repo = _create_repo(tmp_path, {
            "requirements.txt": "celery==5.3\n",
            "tasks.py": (
                "from celery import shared_task\n\n"
                "@shared_task\n"
                "def send_email():\n"
                "    pass\n"
            ),
        })
        detector = RootDetector(repo)
        roots = detector._find_framework_roots()
        assert any("send_email" in r for r in roots)

    def test_detects_nestjs_controllers(self, tmp_path):
        repo = _create_repo(tmp_path, {
            "package.json": '{"dependencies": {"@nestjs/core": "^10.0.0"}}',
            "src/orders.controller.ts": (
                "@Controller('orders')\n"
                "export class OrdersController {\n"
                "  @Get('/')\n"
                "  findAll() { return []; }\n"
                "}\n"
            ),
        })
        detector = RootDetector(repo)
        roots = detector._find_framework_roots()
        assert any("OrdersController" in r for r in roots)

    def test_detects_nextjs_pages(self, tmp_path):
        repo = _create_repo(tmp_path, {
            "package.json": '{"dependencies": {"next": "^14.0.0"}}',
            "app/page.tsx": "export default function Home() { return <div/>; }\n",
        })
        detector = RootDetector(repo)
        roots = detector._find_framework_roots()
        assert any("default" in r for r in roots)

    def test_ignores_uninstalled_frameworks(self, tmp_path):
        repo = _create_repo(tmp_path, {
            "requirements.txt": "requests==2.31\n",
            "views.py": (
                "class OrderView:\n"  # No Django base class
                "    pass\n"
            ),
        })
        detector = RootDetector(repo)
        roots = detector._find_framework_roots()
        assert len(roots) == 0


# ─── Tier 3: Declared Roots ───────────────────────────────────────

class TestDeclaredRoots:
    def test_loads_roots_yml(self, tmp_path):
        repo = _create_repo(tmp_path, {
            ".github/dead-code-agent/roots.yml": (
                "always_live:\n"
                "  - src.rpc.handlers.*\n"
                "  - src.api.health_check\n"
            ),
        })
        detector = RootDetector(repo)
        roots = detector._find_declared_roots()
        assert "src.rpc.handlers.*" in roots
        assert "src.api.health_check" in roots

    def test_empty_roots_yml(self, tmp_path):
        repo = _create_repo(tmp_path, {
            ".github/dead-code-agent/roots.yml": "always_live:\n",
        })
        detector = RootDetector(repo)
        roots = detector._find_declared_roots()
        assert len(roots) == 0

    def test_missing_roots_yml(self, tmp_path):
        detector = RootDetector(tmp_path)
        roots = detector._find_declared_roots()
        assert len(roots) == 0


# ─── Glob Matching ────────────────────────────────────────────────

class TestGlobMatching:
    def test_exact_match(self, tmp_path):
        detector = RootDetector(tmp_path)
        roots = {"src.api.health_check"}
        assert detector.is_root("src.api.health_check", roots)
        assert not detector.is_root("src.api.other", roots)

    def test_wildcard_match(self, tmp_path):
        detector = RootDetector(tmp_path)
        roots = {"src.rpc.handlers.*"}
        assert detector.is_root("src.rpc.handlers.OrderHandler", roots)
        assert not detector.is_root("src.rpc.internal.secret", roots)

    def test_double_wildcard(self, tmp_path):
        detector = RootDetector(tmp_path)
        roots = {"src.plugins.**"}
        # fnmatch ** acts like * for single-level
        assert detector.is_root("src.plugins.auth", roots)


# ─── Full Integration ─────────────────────────────────────────────

class TestFullIntegration:
    def test_all_tiers_combined(self, tmp_path):
        repo = _create_repo(tmp_path, {
            "requirements.txt": "django==4.2\ncelery==5.3\n",
            "app.py": 'if __name__ == "__main__":\n    pass\n',
            "views.py": (
                "from django.views import View\n\n"
                "class MyView(View):\n    pass\n"
            ),
            "tasks.py": (
                "from celery import shared_task\n\n"
                "@shared_task\n"
                "def background_job():\n    pass\n"
            ),
            ".github/dead-code-agent/roots.yml": (
                "always_live:\n  - src.rpc.handler\n"
            ),
        })
        detector = RootDetector(repo)
        roots_by_tier = detector.find_all_roots()

        # Tier 1: main block
        assert len(roots_by_tier["tier1_explicit"]) > 0

        # Tier 2: Django + Celery
        assert len(roots_by_tier["tier2_framework"]) > 0

        # Tier 3: Declared
        assert "src.rpc.handler" in roots_by_tier["tier3_declared"]

        # All tiers combined
        all_roots = detector.get_root_set()
        assert len(all_roots) >= 3  # At least 3 roots from all tiers
