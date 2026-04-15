"""
Shared data models for the Dead Code Removal Agent.

These models flow through every stage:
  Layer 1/2 (detection) → Safety Gate → Deletion Planner → PR Generator → Slack

All models support JSON serialization for artifact passing between GHA jobs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional


# ─────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────

class DeletionType(Enum):
    """
    Classifies the complexity of a deletion.
    Only PURE_DELETION, INLINE_REMOVAL, and CLUSTER_DELETION are auto-PR eligible.
    REQUIRES_REFACTOR always escalates to humans.
    """
    PURE_DELETION    = "pure_deletion"       # File/symbol deleted, no references remain
    INLINE_REMOVAL   = "inline_removal"      # Dead code removed from inside a live file
    CLUSTER_DELETION = "cluster_deletion"    # Multiple symbols deleted together
    REQUIRES_REFACTOR = "requires_refactor"  # Deletion needs call-site cleanup → human only


# ─────────────────────────────────────────────
# Core Data Models
# ─────────────────────────────────────────────

@dataclass
class DeadSymbol:
    """
    A single confirmed-dead code symbol (function, class, method, constant).
    """
    fqn: str                                  # Fully qualified name, e.g. "src.payments.legacy.OldProcessor"
    file_path: str                            # Relative path from repo root
    start_line: int                           # 1-indexed start line in file
    end_line: int                             # 1-indexed end line (inclusive)
    confidence: float                         # 0.0 – 1.0 combined confidence score
    evidence: list[dict] = field(default_factory=list)  # List of evidence dicts
    zombie_cluster_id: Optional[str] = None   # ID of the cluster this symbol belongs to
    deletion_type: DeletionType = DeletionType.PURE_DELETION

    def to_dict(self) -> dict:
        d = asdict(self)
        d["deletion_type"] = self.deletion_type.value
        return d

    @classmethod
    def from_dict(cls, data: dict) -> DeadSymbol:
        data = data.copy()
        data["deletion_type"] = DeletionType(data.get("deletion_type", "pure_deletion"))
        return cls(**data)


@dataclass
class ZombieCluster:
    """
    A group of dead symbols that are only reachable from each other (not from any live entry point).
    Each cluster becomes exactly one PR.
    """
    cluster_id: str                           # Deterministic hash-based ID
    symbols: list[DeadSymbol]
    combined_confidence: float                # Min or weighted avg across symbols
    files_affected: list[str] = field(default_factory=list)
    deletion_plan: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "cluster_id": self.cluster_id,
            "symbols": [s.to_dict() for s in self.symbols],
            "combined_confidence": self.combined_confidence,
            "files_affected": self.files_affected,
            "deletion_plan": self.deletion_plan,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ZombieCluster:
        data = data.copy()
        data["symbols"] = [DeadSymbol.from_dict(s) for s in data.get("symbols", [])]
        return cls(**data)


@dataclass
class PRResult:
    """
    Outcome of attempting to generate a PR for a cluster.
    Serialized to JSON and passed between GHA jobs.
    """
    success: bool
    pr_url: Optional[str]
    branch_name: str
    symbols_deleted: list[str]
    files_modified: list[str]
    confidence: float = 0.0
    title: str = ""
    error: Optional[str] = None
    requires_human_review: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> PRResult:
        return cls(**data)


# ─────────────────────────────────────────────
# Serialization Helpers
# ─────────────────────────────────────────────

def load_clusters_from_json(path: str | Path) -> list[ZombieCluster]:
    """Load a list of ZombieClusters from a JSON file."""
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict):
        data = [data]
    return [ZombieCluster.from_dict(c) for c in data]


def save_pr_result(result: PRResult, path: str | Path) -> None:
    """Write a PRResult to a JSON file."""
    Path(path).write_text(json.dumps(result.to_dict(), indent=2))


def load_pr_result(path: str | Path) -> PRResult:
    """Read a PRResult from a JSON file."""
    return PRResult.from_dict(json.loads(Path(path).read_text()))
