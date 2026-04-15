"""
Adversarial Judge — LLM as defense attorney for the code.

The LLM's job is NOT to confirm code is dead. Its job is to find reasons
the code might be ALIVE that static tools would miss.

If the LLM hallucinates a reason code is alive → we keep dead code (safe).
If we ask it to confirm death and it hallucinates → we break production (catastrophic).

The prompt is designed to:
  1. Present tool-gathered evidence to the LLM (not raw code)
  2. Ask it to challenge the findings adversarially
  3. Force structured output via a strict JSON schema
  4. Cap its influence to "weak" weight in the scoring formula

This module also implements pattern proposal: when the LLM spots a recurring
dynamic dispatch mechanism, it can propose it for learned_patterns.yml.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from graph_builder import GraphNode
from scoring import evidence_llm_assessment

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# Prompt Templates
# ─────────────────────────────────────────────

ADVERSARIAL_SYSTEM_PROMPT = """\
You are a code defense attorney. Your job is to find reasons why code that \
appears dead might actually be alive. You are adversarial — you argue AGAINST \
deletion, not for it.

You will receive:
1. A code symbol (function, class, or method) that static analysis flagged as dead
2. The code itself
3. Evidence from deterministic tools (ripgrep, git blame, AST analysis)

Your task:
- Identify mechanisms that could make this code reachable despite having zero \
  static references (e.g., reflection, dynamic dispatch, string-based imports, \
  framework conventions, webhook handlers, plugin systems)
- If you find such mechanisms, describe them concisely
- If the code genuinely appears dead, say so — do not invent false reasons

CRITICAL RULES:
- You MUST use the provided tool to structure your response
- You MUST NOT hallucinate evidence. If you don't see a mechanism, don't invent one
- Your assessment has WEAK influence on the final decision — tools decide, you advise
"""

ADVERSARIAL_USER_PROMPT = """\
## Symbol Under Review

**FQN:** {fqn}
**File:** {file_path}:{start_line}-{end_line}
**Kind:** {kind}

## Code

```
{code_snippet}
```

## Tool Evidence (Deterministic)

{evidence_summary}

## Your Task

As defense attorney for this code, identify any mechanism that could make it \
reachable despite the tool evidence suggesting it is dead. Consider:

1. Dynamic dispatch (getattr, reflection, eval, string imports)
2. Framework conventions (decorators, naming patterns, config references)
3. External consumers (SDK exports, API contracts, webhook receivers)
4. Build/deploy tooling (scripts referenced in CI, Dockerfiles, Makefiles)
5. Serialization (pickle, JSON schema references, ORM mappings)

If you find a credible mechanism, describe it. If the code is genuinely dead, \
say so clearly.
"""


# ─────────────────────────────────────────────
# Response Schema (for structured output)
# ─────────────────────────────────────────────

ASSESSMENT_TOOL = {
    "name": "submit_assessment",
    "description": "Submit your adversarial assessment of whether the code might be alive.",
    "input_schema": {
        "type": "object",
        "properties": {
            "alive_mechanisms": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "mechanism": {
                            "type": "string",
                            "description": "The specific mechanism (e.g., 'reflection via getattr')",
                        },
                        "evidence_in_code": {
                            "type": "string",
                            "description": "Exact code or pattern that suggests this mechanism",
                        },
                        "credibility": {
                            "type": "string",
                            "enum": ["high", "medium", "low"],
                            "description": "How credible is this mechanism based on the code",
                        },
                    },
                    "required": ["mechanism", "evidence_in_code", "credibility"],
                },
                "description": "List of mechanisms that could make this code alive",
            },
            "overall_assessment": {
                "type": "string",
                "enum": ["likely_dead", "possibly_alive", "likely_alive"],
                "description": "Your overall assessment",
            },
            "reasoning": {
                "type": "string",
                "description": "Brief reasoning for your assessment (max 200 chars)",
            },
            "proposed_pattern": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": [
                            "decorator_root",
                            "class_inheritance_root",
                            "function_call_root",
                            "config_reference_root",
                        ],
                    },
                    "match": {
                        "type": "string",
                        "description": "The decorator/class/function name to match",
                    },
                    "language": {
                        "type": "string",
                        "enum": ["python", "typescript"],
                    },
                },
                "required": ["type", "match", "language"],
                "description": "If you identified a recurring framework pattern, propose it here. Leave null if none found.",
            },
        },
        "required": ["alive_mechanisms", "overall_assessment", "reasoning"],
    },
}


# ─────────────────────────────────────────────
# Judge
# ─────────────────────────────────────────────

class AdversarialJudge:
    """
    Invokes the LLM as a defense attorney for code flagged as dead.

    Returns structured evidence dicts compatible with the scoring module.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 1024,
    ):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.model = model
        self.max_tokens = max_tokens

    def assess_symbol(
        self,
        node: GraphNode,
        code_snippet: str,
        tool_evidence: list[dict],
    ) -> dict:
        """
        Run adversarial assessment on a single symbol.

        Returns an evidence dict compatible with scoring.py:
            {
                "signal": "llm_adversarial_review",
                "detail": "...",
                "weight": "weak",
                "source": "llm_judge",
                "alive_mechanisms": 0,
            }

        If the API call fails, returns a neutral assessment (zero influence).
        """
        if not self.api_key:
            logger.warning("No ANTHROPIC_API_KEY — skipping LLM assessment for %s", node.fqn)
            return evidence_llm_assessment("Skipped — no API key", 0)

        # Build the prompt
        evidence_summary = self._format_evidence(tool_evidence)
        user_prompt = ADVERSARIAL_USER_PROMPT.format(
            fqn=node.fqn,
            file_path=node.file_path,
            start_line=node.start_line,
            end_line=node.end_line,
            kind=node.kind,
            code_snippet=code_snippet[:3000],  # Cap code length
            evidence_summary=evidence_summary,
        )

        try:
            response = self._call_api(user_prompt)
            return self._parse_response(response, node.fqn)
        except Exception as exc:
            logger.warning("LLM assessment failed for %s: %s", node.fqn, exc)
            return evidence_llm_assessment(f"Assessment failed: {exc}", 0)

    def assess_cluster(
        self,
        nodes: list[GraphNode],
        code_snippets: dict[str, str],
        tool_evidence: dict[str, list[dict]],
    ) -> tuple[list[dict], list[dict]]:
        """
        Assess all symbols in a cluster.

        Returns:
            (assessments, proposed_patterns)
            - assessments: list of evidence dicts, one per symbol
            - proposed_patterns: list of pattern proposals for learned_patterns.yml
        """
        assessments = []
        proposed_patterns = []

        for node in nodes:
            snippet = code_snippets.get(node.fqn, "# Code not available")
            evidence = tool_evidence.get(node.fqn, [])

            result = self.assess_symbol(node, snippet, evidence)
            assessments.append(result)

            # Check for pattern proposals
            if hasattr(result, "get") and result.get("_proposed_pattern"):
                proposed_patterns.append(result.pop("_proposed_pattern"))

        return assessments, proposed_patterns

    # ─────────────────────────────────────────────
    # API Call
    # ─────────────────────────────────────────────

    def _call_api(self, user_prompt: str) -> dict:
        """Call the Anthropic Messages API with tool use for structured output."""
        import httpx

        response = httpx.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": self.model,
                "max_tokens": self.max_tokens,
                "system": ADVERSARIAL_SYSTEM_PROMPT,
                "tools": [ASSESSMENT_TOOL],
                "tool_choice": {"type": "tool", "name": "submit_assessment"},
                "messages": [
                    {"role": "user", "content": user_prompt},
                ],
            },
            timeout=30,
        )
        response.raise_for_status()
        return response.json()

    # ─────────────────────────────────────────────
    # Response Parsing
    # ─────────────────────────────────────────────

    def _parse_response(self, response: dict, fqn: str) -> dict:
        """Parse the structured API response into an evidence dict."""
        # Extract tool_use content block
        tool_input = None
        for block in response.get("content", []):
            if block.get("type") == "tool_use" and block.get("name") == "submit_assessment":
                tool_input = block.get("input", {})
                break

        if not tool_input:
            logger.warning("No structured response for %s", fqn)
            return evidence_llm_assessment("No structured response received", 0)

        # Count credible alive mechanisms
        mechanisms = tool_input.get("alive_mechanisms", [])
        credible_count = sum(
            1 for m in mechanisms
            if m.get("credibility") in ("high", "medium")
        )

        reasoning = tool_input.get("reasoning", "")
        assessment = tool_input.get("overall_assessment", "likely_dead")

        result = evidence_llm_assessment(
            reasoning=f"[{assessment}] {reasoning}",
            alive_mechanisms_found=credible_count,
        )

        # Attach pattern proposal if present
        proposed = tool_input.get("proposed_pattern")
        if proposed and proposed.get("match"):
            result["_proposed_pattern"] = {
                "type": proposed["type"],
                "match": proposed["match"],
                "language": proposed["language"],
                "discovered_for": fqn,
                "status": "proposed",
            }

        return result

    @staticmethod
    def _format_evidence(evidence: list[dict]) -> str:
        """Format tool evidence for the prompt."""
        if not evidence:
            return "No tool evidence available."

        lines = []
        for e in evidence:
            signal = e.get("signal", "unknown")
            detail = e.get("detail", "")
            weight = e.get("weight", "unknown")
            source = e.get("source", "unknown")
            lines.append(f"- **{signal}** [{weight}] (source: {source}): {detail}")
        return "\n".join(lines)
