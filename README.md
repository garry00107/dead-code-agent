# 🧟 Dead Code Agent

A production-grade, deterministic dead code removal system that combines static analysis, graph theory, and adversarial LLM reasoning to safely identify and remove dead code from enterprise codebases.

## Architecture

```
Layer 1 (Static Analysis)     Layer 2 (Agent)           Layer 3 (PR Generation)
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│ Root Detection (4T) │    │ Adversarial LLM     │    │ Safety Gates (6)    │
│ Graph Builder (AST) │───▶│ Evidence Scoring    │───▶│ Deletion Planner    │
│ Tarjan's SCC        │    │ Confidence Calc     │    │ Smoke Tests → PR    │
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
         100% deterministic      LLM capped at 0.05 weight      6 hard blockers
```

## Key Design Principles

- **Tools discover, LLM challenges** — the LLM is a defense attorney for the code, not a judge
- **Asymmetric risk** — over-classifying roots (keeping dead code) is safe; missing a root breaks production
- **Atomic PRs** — one zombie cluster per PR, never a mass deletion
- **Git-native observability** — shadow reports committed to the repo, triage via `git blame`

## Quick Start

```bash
# Run tests
pip install pytest
python -m pytest .github/dead-code-agent/tests/ -v

# Self-test (point the agent at itself)
python .github/dead-code-agent/dry_run_self.py

# Copy .env.example to .env and add your API keys
cp .env.example .env
```

## Shadow Mode (Calibration)

The agent starts in `dry_run: true` mode. It generates reports without opening PRs.

```bash
# Triage a cluster from a shadow report
python .github/dead-code-agent/triage_tracker.py triage \
  --report .github/dead-code-agent/reports/run-YYYYMMDD.json \
  --cluster <cluster-id> \
  --status true_positive \
  --reviewer your-handle

# View the trend dashboard
python .github/dead-code-agent/triage_tracker.py dashboard
```

Once the false-positive rate hits 0% across 3+ runs, flip `dry_run` to `false` in the workflow.

## CI/CD

The GitHub Actions workflow (`.github/workflows/dead-code-agent.yml`) runs weekly and supports:
- Manual trigger via `workflow_dispatch`
- Shadow mode reporting
- Auto-PR generation (when `dry_run: false`)
- Slack notifications

## Test Results

```
174 passed in 0.49s
```

## License

MIT
