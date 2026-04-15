"""
Framework Detectors — Tier 2 root detection via deterministic AST analysis.

Each detector is a plugin that identifies framework-specific entry points
(routes, DI bindings, event handlers) that static call-graph analysis
cannot detect because they're invoked by the framework, not by user code.

Supported frameworks:
  Python:     Django, FastAPI, Flask, Celery
  TypeScript: NestJS, Express, Next.js

Design:
  - Every detector produces a set of fully-qualified symbol names (FQNs)
  - Detection is 100% deterministic — AST parsing, no LLM
  - Each plugin is ~50-100 lines
  - False positives (over-classifying roots) are safe
  - False negatives (missing roots) cause production deletions — unacceptable
"""

from __future__ import annotations

import ast
import json
import logging
import os
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Base class
# ─────────────────────────────────────────────

class FrameworkDetector(ABC):
    """Base class for all framework detectors."""

    name: str = "unknown"
    language: str = "unknown"

    @abstractmethod
    def detect_framework(self, repo_path: Path) -> bool:
        """Return True if this framework is present in the repo."""
        ...

    @abstractmethod
    def find_roots(self, repo_path: Path) -> set[str]:
        """Return FQNs of all symbols that are framework entry points."""
        ...

    def _python_files(self, repo_path: Path, scope: str = "**/*.py") -> list[Path]:
        """Glob for Python files, excluding common non-source dirs."""
        exclude = {".venv", "venv", "node_modules", ".git", "__pycache__", ".tox", "dist"}
        results = []
        for p in repo_path.glob(scope):
            if any(part in exclude for part in p.parts):
                continue
            results.append(p)
        return results

    def _ts_files(self, repo_path: Path, scope: str = "**/*.ts") -> list[Path]:
        """Glob for TypeScript files."""
        exclude = {"node_modules", ".git", "dist", "build", ".next"}
        results = []
        for pattern in [scope, scope.replace(".ts", ".tsx")]:
            for p in repo_path.glob(pattern):
                if any(part in exclude for part in p.parts):
                    continue
                results.append(p)
        return results

    def _path_to_fqn(self, file_path: Path, repo_path: Path, symbol_name: str) -> str:
        """Convert a file path + symbol name to a dotted FQN."""
        rel = file_path.relative_to(repo_path)
        module = str(rel).replace(os.sep, ".").replace("/", ".")
        # Strip .py / .ts extension
        for ext in (".py", ".ts", ".tsx", ".js", ".jsx"):
            if module.endswith(ext):
                module = module[: -len(ext)]
                break
        return f"{module}.{symbol_name}"


# ─────────────────────────────────────────────
# Python Framework Detectors
# ─────────────────────────────────────────────

class DjangoDetector(FrameworkDetector):
    """
    Detects Django roots: views, models, serializers, management commands,
    signal handlers, admin registrations, URL patterns.
    """

    name = "django"
    language = "python"

    # Classes inheriting from these are roots
    ROOT_BASE_CLASSES = {
        # Views
        "View", "TemplateView", "ListView", "DetailView", "CreateView",
        "UpdateView", "DeleteView", "FormView", "RedirectView",
        "APIView", "ViewSet", "ModelViewSet", "ReadOnlyModelViewSet",
        "GenericAPIView", "GenericViewSet",
        # Models
        "Model", "AbstractUser", "AbstractBaseUser",
        # Serializers
        "Serializer", "ModelSerializer", "HyperlinkedModelSerializer",
        # Admin
        "ModelAdmin", "TabularInline", "StackedInline",
        # Commands
        "BaseCommand",
        # Middleware
        "MiddlewareMixin",
        # Forms
        "Form", "ModelForm",
    }

    # Decorators that make functions roots
    ROOT_DECORATORS = {
        "receiver",           # signal handlers
        "register",           # admin.register
        "action",             # DRF viewset actions
        "api_view",           # DRF function-based views
        "login_required",     # implicitly marks as a view
        "permission_required",
    }

    def detect_framework(self, repo_path: Path) -> bool:
        # Check for Django in requirements or settings.py
        for req_file in ["requirements.txt", "Pipfile", "pyproject.toml"]:
            req_path = repo_path / req_file
            if req_path.exists():
                content = req_path.read_text(errors="replace").lower()
                if "django" in content:
                    return True
        # Check for manage.py
        if (repo_path / "manage.py").exists():
            return True
        return False

    def find_roots(self, repo_path: Path) -> set[str]:
        roots = set()
        for py_file in self._python_files(repo_path):
            try:
                tree = ast.parse(py_file.read_text(errors="replace"))
            except SyntaxError:
                continue

            for node in ast.walk(tree):
                # Class-based roots (inheritance)
                if isinstance(node, ast.ClassDef):
                    for base in node.bases:
                        base_name = self._get_name(base)
                        if base_name in self.ROOT_BASE_CLASSES:
                            roots.add(self._path_to_fqn(py_file, repo_path, node.name))
                            # Also add all methods
                            for item in node.body:
                                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                    roots.add(self._path_to_fqn(
                                        py_file, repo_path, f"{node.name}.{item.name}"
                                    ))
                            break

                # Decorator-based roots
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    for dec in node.decorator_list:
                        dec_name = self._get_name(dec)
                        if dec_name in self.ROOT_DECORATORS:
                            roots.add(self._path_to_fqn(py_file, repo_path, node.name))
                            break

        return roots

    @staticmethod
    def _get_name(node: ast.AST) -> str:
        """Extract the simple name from a decorator or base class node."""
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        if isinstance(node, ast.Call):
            return DjangoDetector._get_name(node.func)
        return ""


class FastAPIDetector(FrameworkDetector):
    """
    Detects FastAPI/Starlette roots: route decorators, dependency injection,
    startup/shutdown events, middleware.
    """

    name = "fastapi"
    language = "python"

    ROUTE_DECORATORS = {
        "get", "post", "put", "patch", "delete", "options", "head", "trace",
        "api_route", "websocket",
    }

    ROOT_DECORATORS = {
        "on_event",       # startup/shutdown handlers
        "middleware",     # middleware functions
        "exception_handler",
    }

    def detect_framework(self, repo_path: Path) -> bool:
        for req_file in ["requirements.txt", "Pipfile", "pyproject.toml"]:
            req_path = repo_path / req_file
            if req_path.exists():
                content = req_path.read_text(errors="replace").lower()
                if "fastapi" in content or "starlette" in content:
                    return True
        return False

    def find_roots(self, repo_path: Path) -> set[str]:
        roots = set()
        for py_file in self._python_files(repo_path):
            try:
                tree = ast.parse(py_file.read_text(errors="replace"))
            except SyntaxError:
                continue

            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                for dec in node.decorator_list:
                    dec_name = self._get_decorator_name(dec)
                    if dec_name in self.ROUTE_DECORATORS | self.ROOT_DECORATORS:
                        roots.add(self._path_to_fqn(py_file, repo_path, node.name))
                        break

                    # Handle app.get("/path") style
                    if isinstance(dec, ast.Call):
                        func_name = self._get_decorator_name(dec.func) if hasattr(dec, 'func') else ""
                        if func_name in self.ROUTE_DECORATORS:
                            roots.add(self._path_to_fqn(py_file, repo_path, node.name))
                            break

        return roots

    @staticmethod
    def _get_decorator_name(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        if isinstance(node, ast.Call) and hasattr(node, 'func'):
            return FastAPIDetector._get_decorator_name(node.func)
        return ""


class FlaskDetector(FrameworkDetector):
    """
    Detects Flask roots: route decorators, CLI commands, error handlers,
    before/after request hooks.
    """

    name = "flask"
    language = "python"

    ROOT_DECORATORS = {
        "route", "get", "post", "put", "patch", "delete",
        "before_request", "after_request", "teardown_request",
        "before_first_request",
        "errorhandler",
        "cli.command", "command",
    }

    def detect_framework(self, repo_path: Path) -> bool:
        for req_file in ["requirements.txt", "Pipfile", "pyproject.toml"]:
            req_path = repo_path / req_file
            if req_path.exists():
                content = req_path.read_text(errors="replace").lower()
                if "flask" in content:
                    return True
        return False

    def find_roots(self, repo_path: Path) -> set[str]:
        roots = set()
        for py_file in self._python_files(repo_path):
            try:
                tree = ast.parse(py_file.read_text(errors="replace"))
            except SyntaxError:
                continue

            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                for dec in node.decorator_list:
                    dec_name = self._get_name(dec)
                    if dec_name in self.ROOT_DECORATORS:
                        roots.add(self._path_to_fqn(py_file, repo_path, node.name))
                        break

        return roots

    @staticmethod
    def _get_name(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        if isinstance(node, ast.Call):
            return FlaskDetector._get_name(node.func)
        return ""


class CeleryDetector(FrameworkDetector):
    """
    Detects Celery roots: task decorators, beat schedule entries.
    """

    name = "celery"
    language = "python"

    ROOT_DECORATORS = {"task", "shared_task", "periodic_task"}

    def detect_framework(self, repo_path: Path) -> bool:
        for req_file in ["requirements.txt", "Pipfile", "pyproject.toml"]:
            req_path = repo_path / req_file
            if req_path.exists():
                content = req_path.read_text(errors="replace").lower()
                if "celery" in content:
                    return True
        return False

    def find_roots(self, repo_path: Path) -> set[str]:
        roots = set()
        for py_file in self._python_files(repo_path):
            try:
                tree = ast.parse(py_file.read_text(errors="replace"))
            except SyntaxError:
                continue

            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                for dec in node.decorator_list:
                    dec_name = self._get_name(dec)
                    if dec_name in self.ROOT_DECORATORS:
                        roots.add(self._path_to_fqn(py_file, repo_path, node.name))
                        break

        return roots

    @staticmethod
    def _get_name(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        if isinstance(node, ast.Call):
            return CeleryDetector._get_name(node.func)
        return ""


# ─────────────────────────────────────────────
# TypeScript Framework Detectors
# ─────────────────────────────────────────────

class NestJSDetector(FrameworkDetector):
    """
    Detects NestJS roots: controllers, injectables, modules, guards,
    interceptors, pipes, exception filters.

    Uses regex since we can't use Python's ast module for TypeScript.
    Patterns are conservative (may over-match) — safe by design.
    """

    name = "nestjs"
    language = "typescript"

    # Decorators that mark a class as a root
    ROOT_CLASS_DECORATORS = {
        "Controller", "Injectable", "Module",
        "Guard", "UseGuards",
        "Interceptor", "UseInterceptors",
        "Pipe", "UsePipes",
        "Catch",  # exception filters
    }

    # Method decorators that mark methods as roots
    ROOT_METHOD_DECORATORS = {
        "Get", "Post", "Put", "Patch", "Delete", "Options", "Head", "All",
        "Sse", "MessagePattern", "EventPattern",
        "Cron", "Interval", "Timeout",  # schedule module
    }

    # Regex patterns
    _CLASS_DECORATOR_RE = re.compile(
        r"@(" + "|".join(ROOT_CLASS_DECORATORS) + r")\s*\([^)]*\)\s*\n\s*(?:export\s+)?class\s+(\w+)",
        re.MULTILINE,
    )
    _METHOD_DECORATOR_RE = re.compile(
        r"@(" + "|".join(ROOT_METHOD_DECORATORS) + r")\s*\([^)]*\)\s*\n\s*(?:async\s+)?(\w+)\s*\(",
        re.MULTILINE,
    )

    def detect_framework(self, repo_path: Path) -> bool:
        pkg_json = repo_path / "package.json"
        if pkg_json.exists():
            try:
                pkg = json.loads(pkg_json.read_text())
                deps = {**pkg.get("dependencies", {}), **pkg.get("devDependencies", {})}
                return "@nestjs/core" in deps or "@nestjs/common" in deps
            except (json.JSONDecodeError, KeyError):
                pass
        return False

    def find_roots(self, repo_path: Path) -> set[str]:
        roots = set()
        for ts_file in self._ts_files(repo_path):
            try:
                content = ts_file.read_text(errors="replace")
            except OSError:
                continue

            # Class-level decorators
            for match in self._CLASS_DECORATOR_RE.finditer(content):
                class_name = match.group(2)
                roots.add(self._path_to_fqn(ts_file, repo_path, class_name))

            # Method-level decorators
            for match in self._METHOD_DECORATOR_RE.finditer(content):
                method_name = match.group(2)
                roots.add(self._path_to_fqn(ts_file, repo_path, method_name))

        return roots


class ExpressDetector(FrameworkDetector):
    """
    Detects Express.js roots: route handlers registered via app.get/post/etc
    and router.get/post/etc.
    """

    name = "express"
    language = "typescript"

    _ROUTE_RE = re.compile(
        r"(?:app|router)\s*\.\s*(get|post|put|patch|delete|use|all|options|head)"
        r"\s*\(\s*['\"]([^'\"]+)['\"]",
        re.MULTILINE,
    )

    # Named handler functions passed to routes
    _HANDLER_RE = re.compile(
        r"(?:app|router)\s*\.\s*(?:get|post|put|patch|delete|use|all)\s*\([^,]+,\s*(\w+)\s*\)",
        re.MULTILINE,
    )

    def detect_framework(self, repo_path: Path) -> bool:
        pkg_json = repo_path / "package.json"
        if pkg_json.exists():
            try:
                pkg = json.loads(pkg_json.read_text())
                deps = {**pkg.get("dependencies", {}), **pkg.get("devDependencies", {})}
                return "express" in deps
            except (json.JSONDecodeError, KeyError):
                pass
        return False

    def find_roots(self, repo_path: Path) -> set[str]:
        roots = set()
        for ts_file in self._ts_files(repo_path, "**/*.ts"):
            try:
                content = ts_file.read_text(errors="replace")
            except OSError:
                continue

            # Named handlers passed to route registrations
            for match in self._HANDLER_RE.finditer(content):
                handler_name = match.group(1)
                roots.add(self._path_to_fqn(ts_file, repo_path, handler_name))

        # Also check .js files
        for js_file in repo_path.glob("**/*.js"):
            if any(part in {"node_modules", ".git", "dist"} for part in js_file.parts):
                continue
            try:
                content = js_file.read_text(errors="replace")
            except OSError:
                continue
            for match in self._HANDLER_RE.finditer(content):
                handler_name = match.group(1)
                roots.add(self._path_to_fqn(js_file, repo_path, handler_name))

        return roots


class NextJSDetector(FrameworkDetector):
    """
    Detects Next.js roots: page components, API routes, layout/loading files,
    server actions, metadata exports.
    """

    name = "nextjs"
    language = "typescript"

    # Files that are always roots by convention in Next.js
    CONVENTION_FILES = {
        "page", "layout", "loading", "error", "not-found",
        "template", "default", "route", "middleware",
        "global-error", "instrumentation",
    }

    # Exports that are roots
    ROOT_EXPORTS = {
        "default", "generateStaticParams", "generateMetadata",
        "GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS",
        "getServerSideProps", "getStaticProps", "getStaticPaths",
    }

    def detect_framework(self, repo_path: Path) -> bool:
        pkg_json = repo_path / "package.json"
        if pkg_json.exists():
            try:
                pkg = json.loads(pkg_json.read_text())
                deps = {**pkg.get("dependencies", {}), **pkg.get("devDependencies", {})}
                return "next" in deps
            except (json.JSONDecodeError, KeyError):
                pass
        return False

    def find_roots(self, repo_path: Path) -> set[str]:
        roots = set()

        # Convention-based files in app/ or pages/ directories
        for dir_name in ["app", "pages", "src/app", "src/pages"]:
            base_dir = repo_path / dir_name
            if not base_dir.exists():
                continue
            for ext in [".tsx", ".ts", ".jsx", ".js"]:
                for f in base_dir.rglob(f"*{ext}"):
                    stem = f.stem
                    if stem in self.CONVENTION_FILES:
                        # The entire file is a root — mark the default export
                        roots.add(self._path_to_fqn(f, repo_path, "default"))

                    # Check for root exports inside the file
                    try:
                        content = f.read_text(errors="replace")
                    except OSError:
                        continue
                    for export_name in self.ROOT_EXPORTS:
                        # Match: export default, export function X, export const X, export async function X
                        pattern = (
                            rf"export\s+(?:default|(?:async\s+)?function\s+{export_name}"
                            rf"|const\s+{export_name})"
                        )
                        if re.search(pattern, content):
                            roots.add(self._path_to_fqn(f, repo_path, export_name))

        return roots


# ─────────────────────────────────────────────
# Learned Pattern Detector (Tier 2.5)
# ─────────────────────────────────────────────

class LearnedPatternDetector(FrameworkDetector):
    """
    Applies confirmed patterns from learned_patterns.yml using
    parameterized AST searches — NOT regex against raw source.

    Supported types:
      - decorator_root:          AST-based decorator search (Python)
      - class_inheritance_root:  AST-based base class search (Python)
      - function_call_root:      AST-based call-arg search (Python)
      - config_reference_root:   String search in config files (YAML/JSON)

    For TypeScript, falls back to regex-based decorator search since
    Python's ast module can't parse TS.
    """

    name = "learned_patterns"
    language = "multi"

    def __init__(self, patterns_path: Optional[Path] = None):
        self._patterns_path = patterns_path

    def detect_framework(self, repo_path: Path) -> bool:
        path = self._patterns_path or (repo_path / ".github" / "dead-code-agent" / "learned_patterns.yml")
        if not path.exists():
            return False
        try:
            import yaml
        except ImportError:
            # Fall back to simple parsing if PyYAML not available
            content = path.read_text()
            return "status: confirmed" in content
        return True

    def find_roots(self, repo_path: Path) -> set[str]:
        path = self._patterns_path or (repo_path / ".github" / "dead-code-agent" / "learned_patterns.yml")
        if not path.exists():
            return set()

        patterns = self._load_patterns(path)
        roots = set()

        for p in patterns:
            if p.get("status") != "confirmed":
                continue

            ptype = p.get("type", "")
            match_str = p.get("match", "")
            scope = p.get("scope", "**/*.py")
            lang = p.get("language", "python")

            if not match_str:
                continue

            if ptype == "decorator_root":
                if lang == "python":
                    roots |= self._ast_find_decorator(repo_path, match_str, scope)
                else:
                    roots |= self._regex_find_decorator(repo_path, match_str, scope)

            elif ptype == "class_inheritance_root":
                if lang == "python":
                    roots |= self._ast_find_subclasses(repo_path, match_str, scope)

            elif ptype == "function_call_root":
                if lang == "python":
                    roots |= self._ast_find_call_args(repo_path, match_str, scope)

            elif ptype == "config_reference_root":
                roots |= self._find_in_config_files(repo_path, match_str, scope)

        return roots

    def _load_patterns(self, path: Path) -> list[dict]:
        """Load patterns from YAML, with fallback for no PyYAML."""
        try:
            import yaml
            data = yaml.safe_load(path.read_text()) or {}
            return data.get("patterns", []) or []
        except ImportError:
            # Minimal YAML parsing fallback
            logger.warning("PyYAML not installed — skipping learned patterns")
            return []

    def _ast_find_decorator(self, repo_path: Path, decorator_name: str, scope: str) -> set[str]:
        """Find functions/classes decorated with a specific decorator via AST."""
        roots = set()
        for py_file in self._python_files(repo_path, scope):
            try:
                tree = ast.parse(py_file.read_text(errors="replace"))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    for dec in node.decorator_list:
                        if self._decorator_matches(dec, decorator_name):
                            roots.add(self._path_to_fqn(py_file, repo_path, node.name))
                            break
        return roots

    def _ast_find_subclasses(self, repo_path: Path, base_class: str, scope: str) -> set[str]:
        """Find classes inheriting from a specific base class via AST."""
        roots = set()
        for py_file in self._python_files(repo_path, scope):
            try:
                tree = ast.parse(py_file.read_text(errors="replace"))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    for base in node.bases:
                        name = self._get_simple_name(base)
                        if name == base_class:
                            roots.add(self._path_to_fqn(py_file, repo_path, node.name))
                            break
        return roots

    def _ast_find_call_args(self, repo_path: Path, function_name: str, scope: str) -> set[str]:
        """Find symbols passed as arguments to a registration function via AST."""
        roots = set()
        for py_file in self._python_files(repo_path, scope):
            try:
                tree = ast.parse(py_file.read_text(errors="replace"))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    call_name = self._get_simple_name(node.func)
                    if call_name == function_name:
                        # Every named argument is a root
                        for arg in node.args:
                            name = self._get_simple_name(arg)
                            if name:
                                roots.add(self._path_to_fqn(py_file, repo_path, name))
        return roots

    def _find_in_config_files(self, repo_path: Path, symbol_name: str, scope: str) -> set[str]:
        """Find symbol references in config files (YAML, JSON, TOML)."""
        roots = set()
        for ext in ["*.yml", "*.yaml", "*.json", "*.toml"]:
            for f in repo_path.glob(ext):
                try:
                    content = f.read_text(errors="replace")
                except OSError:
                    continue
                if symbol_name in content:
                    # Can't determine FQN from config — add the raw symbol name
                    roots.add(symbol_name)
        return roots

    def _regex_find_decorator(self, repo_path: Path, decorator_name: str, scope: str) -> set[str]:
        """Regex fallback for TypeScript decorator detection."""
        roots = set()
        pattern = re.compile(
            rf"@{re.escape(decorator_name)}\s*\([^)]*\)\s*\n\s*(?:export\s+)?(?:class|async\s+)?(\w+)",
            re.MULTILINE,
        )
        for ts_file in self._ts_files(repo_path, scope):
            try:
                content = ts_file.read_text(errors="replace")
            except OSError:
                continue
            for match in pattern.finditer(content):
                roots.add(self._path_to_fqn(ts_file, repo_path, match.group(1)))
        return roots

    @staticmethod
    def _decorator_matches(dec_node: ast.AST, target: str) -> bool:
        """Check if a decorator AST node matches the target name."""
        name = LearnedPatternDetector._get_simple_name(dec_node)
        return name == target

    @staticmethod
    def _get_simple_name(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        if isinstance(node, ast.Call) and hasattr(node, 'func'):
            return LearnedPatternDetector._get_simple_name(node.func)
        return ""


# ─────────────────────────────────────────────
# Registry — all available detectors
# ─────────────────────────────────────────────

ALL_DETECTORS: list[type[FrameworkDetector]] = [
    DjangoDetector,
    FastAPIDetector,
    FlaskDetector,
    CeleryDetector,
    NestJSDetector,
    ExpressDetector,
    NextJSDetector,
]
