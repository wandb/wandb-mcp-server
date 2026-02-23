# Skills Evaluation Framework

Python-native evaluation framework for MCP skills, built on Weave's `Evaluation` and `Scorer` classes.

## Architecture

```
_evals/
  conftest.py          # Shared fixtures: Weave init, MCP client mock, sample data
  scorers.py           # Reusable Weave Scorer classes
  test_experiment.py   # Evals for experiment-analysis skill
  test_trace.py        # Evals for trace-analyst skill
  test_quickstart.py   # Evals for quickstart skill
  test_failure.py      # Evals for failure-analysis skill
```

## Running Evals

```bash
# All skills
pytest skills/_evals/ -v

# Single skill
pytest skills/_evals/test_experiment.py -v

# With Weave logging (results tracked in W&B)
WANDB_API_KEY=... pytest skills/_evals/ -v
```

## How It Works

Each eval file defines:

1. **Dataset** -- sample inputs representing real user scenarios for the skill
2. **Model function** -- simulates the agent executing the skill workflow
3. **Scorers** -- `weave.Scorer` subclasses that grade skill execution quality

Scorers evaluate:
- **Tool selection** -- did the agent pick the right MCP tool?
- **Workflow ordering** -- did it follow the prescribed step sequence?
- **Output quality** -- is the final answer useful, accurate, complete?
- **Efficiency** -- how many tool calls were needed? (fewer is better)

## Adding a New Eval

1. Create `test_{skill_name}.py` in this directory
2. Define a dataset of 5-10 scenarios as dicts
3. Write a model function that returns the skill's output given a scenario
4. Use scorers from `scorers.py` or define skill-specific ones
5. Wire into `weave.Evaluation` and run with `pytest`

## Design Decisions

- **Python-native over YAML** -- scorers are debuggable Python classes, not config
- **Weave-native** -- all results logged to W&B for visual comparison across versions
- **Pytest-compatible** -- integrates with CI, existing test infrastructure
- **Scorer versioning** -- scorers are `weave.Scorer` subclasses, publishable and versioned
