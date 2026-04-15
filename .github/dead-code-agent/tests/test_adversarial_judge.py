"""
Tests for AdversarialJudge — verifies prompt structure and response parsing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adversarial_judge import (
    AdversarialJudge,
    ADVERSARIAL_SYSTEM_PROMPT,
    ASSESSMENT_TOOL,
)
from graph_builder import GraphNode


# ─── Prompt Design ────────────────────────────────────────────────

class TestPromptDesign:
    def test_system_prompt_is_adversarial(self):
        """The system prompt should instruct the LLM to argue AGAINST deletion."""
        assert "defense attorney" in ADVERSARIAL_SYSTEM_PROMPT.lower()
        assert "against deletion" in ADVERSARIAL_SYSTEM_PROMPT.lower() or \
               "argue against" in ADVERSARIAL_SYSTEM_PROMPT.lower() or \
               "adversarial" in ADVERSARIAL_SYSTEM_PROMPT.lower()

    def test_system_prompt_warns_about_weak_influence(self):
        """LLM should know its assessment has limited weight."""
        assert "weak" in ADVERSARIAL_SYSTEM_PROMPT.lower()

    def test_system_prompt_forbids_hallucination(self):
        assert "hallucinate" in ADVERSARIAL_SYSTEM_PROMPT.lower()


# ─── Tool Schema ──────────────────────────────────────────────────

class TestToolSchema:
    def test_schema_has_required_fields(self):
        props = ASSESSMENT_TOOL["input_schema"]["properties"]
        assert "alive_mechanisms" in props
        assert "overall_assessment" in props
        assert "reasoning" in props

    def test_assessment_is_constrained_enum(self):
        """Overall assessment must be one of 3 values — no free text."""
        assessment_schema = ASSESSMENT_TOOL["input_schema"]["properties"]["overall_assessment"]
        assert "enum" in assessment_schema
        assert set(assessment_schema["enum"]) == {
            "likely_dead", "possibly_alive", "likely_alive",
        }

    def test_credibility_is_constrained(self):
        """Mechanism credibility must be high/medium/low — no free text."""
        mechanism_schema = ASSESSMENT_TOOL["input_schema"]["properties"]["alive_mechanisms"]["items"]
        cred = mechanism_schema["properties"]["credibility"]
        assert "enum" in cred
        assert set(cred["enum"]) == {"high", "medium", "low"}

    def test_proposed_pattern_type_constrained(self):
        """Pattern proposals must use pre-built AST strategies."""
        pattern_schema = ASSESSMENT_TOOL["input_schema"]["properties"]["proposed_pattern"]
        type_prop = pattern_schema["properties"]["type"]
        assert "enum" in type_prop
        expected = {
            "decorator_root", "class_inheritance_root",
            "function_call_root", "config_reference_root",
        }
        assert set(type_prop["enum"]) == expected


# ─── Response Parsing ─────────────────────────────────────────────

class TestResponseParsing:
    def setup_method(self):
        self.judge = AdversarialJudge(api_key="test-key")
        self.sample_node = GraphNode(
            fqn="src.legacy.process",
            file_path="src/legacy.py",
            start_line=10, end_line=25,
            kind="function", name="process",
        )

    def test_parses_likely_dead(self):
        response = {
            "content": [{
                "type": "tool_use",
                "name": "submit_assessment",
                "input": {
                    "alive_mechanisms": [],
                    "overall_assessment": "likely_dead",
                    "reasoning": "No dynamic dispatch found",
                },
            }],
        }
        result = self.judge._parse_response(response, "test.func")
        assert result["signal"] == "llm_adversarial_review"
        assert result["weight"] == "weak"  # Always weak
        assert result["alive_mechanisms"] == 0
        assert "likely_dead" in result["detail"]

    def test_parses_alive_mechanisms(self):
        response = {
            "content": [{
                "type": "tool_use",
                "name": "submit_assessment",
                "input": {
                    "alive_mechanisms": [
                        {
                            "mechanism": "reflection via getattr",
                            "evidence_in_code": "getattr(module, name)",
                            "credibility": "high",
                        },
                        {
                            "mechanism": "string import",
                            "evidence_in_code": "__import__(name)",
                            "credibility": "low",
                        },
                    ],
                    "overall_assessment": "possibly_alive",
                    "reasoning": "Found reflection pattern",
                },
            }],
        }
        result = self.judge._parse_response(response, "test.func")
        # Only high/medium count as credible
        assert result["alive_mechanisms"] == 1

    def test_parses_pattern_proposal(self):
        response = {
            "content": [{
                "type": "tool_use",
                "name": "submit_assessment",
                "input": {
                    "alive_mechanisms": [],
                    "overall_assessment": "possibly_alive",
                    "reasoning": "Custom decorator pattern found",
                    "proposed_pattern": {
                        "type": "decorator_root",
                        "match": "register_handler",
                        "language": "python",
                    },
                },
            }],
        }
        result = self.judge._parse_response(response, "test.func")
        assert "_proposed_pattern" in result
        assert result["_proposed_pattern"]["type"] == "decorator_root"
        assert result["_proposed_pattern"]["match"] == "register_handler"

    def test_no_tool_use_returns_neutral(self):
        response = {"content": [{"type": "text", "text": "I think it's dead"}]}
        result = self.judge._parse_response(response, "test.func")
        assert result["signal"] == "llm_adversarial_review"
        assert result["alive_mechanisms"] == 0

    def test_no_api_key_returns_skipped(self):
        judge = AdversarialJudge(api_key="")
        node = GraphNode(
            fqn="test.func", file_path="test.py",
            start_line=1, end_line=5, kind="function", name="func",
        )
        result = judge.assess_symbol(node, "def func(): pass", [])
        assert "Skipped" in result["detail"]
        assert result["weight"] == "weak"


# ─── Evidence Format ──────────────────────────────────────────────

class TestEvidenceFormat:
    def test_evidence_formatting(self):
        judge = AdversarialJudge(api_key="test")
        evidence = [
            {"signal": "zero_references", "detail": "0 matches", "weight": "strong", "source": "ripgrep"},
            {"signal": "scc_isolated", "detail": "SCC of 2", "weight": "strong", "source": "graph"},
        ]
        formatted = judge._format_evidence(evidence)
        assert "zero_references" in formatted
        assert "strong" in formatted
        assert "ripgrep" in formatted

    def test_empty_evidence(self):
        judge = AdversarialJudge(api_key="test")
        assert "No tool evidence" in judge._format_evidence([])
