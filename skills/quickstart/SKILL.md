---
name: quickstart
description: Instrument a codebase with W&B Weave for LLM observability. Use when the user asks to "add tracing", "instrument my app", "set up Weave", "add observability", "get started with Weave", or "I want to see my LLM calls".
---

# Quickstart / Instrumentation

Guide users from zero to first Weave trace in under 5 minutes.

## When to Use

- User wants to add LLM observability to their app
- User is new to Weave and wants to get started
- User asks how to trace their LLM calls

## Workflow

### Step 1: Detect the Framework

Scan the user's codebase for known frameworks:

| Look for | Framework | Patching |
|----------|-----------|----------|
| `import openai` | OpenAI | Automatic with `weave.init()` |
| `import anthropic` | Anthropic | Automatic with `weave.init()` |
| `from langchain` | LangChain | Automatic with `weave.init()` |
| `import litellm` | LiteLLM | Automatic with `weave.init()` |
| `import instructor` | Instructor | Automatic (patches underlying client) |
| `import google.generativeai` | Google GenAI | Automatic with `weave.init()` |
| Custom LLM wrapper | Custom | Needs `@weave.op()` decorator |

### Step 2: Add Weave Initialization

At the application's entry point (e.g., `main.py`, `app.py`, the file with `if __name__`):

```python
import weave

weave.init("my-project")
```

This single line enables:
- Auto-patching of all supported LLM libraries
- Trace collection to W&B cloud
- Cost and token tracking

### Step 3: Wrap Custom Functions

For any custom logic the user wants to trace:

```python
@weave.op()
def my_pipeline(query: str) -> str:
    """Wrapping with @weave.op() makes this function a traced operation."""
    result = llm_client.chat(messages=[{"role": "user", "content": query}])
    return result.choices[0].message.content
```

### Step 4: Add Evaluation (Optional)

If the user has a dataset and scoring function:

```python
import weave

@weave.op()
def my_scorer(output: str, expected: str) -> dict:
    return {"correct": output.strip() == expected.strip()}

dataset = [
    {"query": "What is 2+2?", "expected": "4"},
    {"query": "Capital of France?", "expected": "Paris"},
]

evaluation = weave.Evaluation(dataset=dataset, scorers=[my_scorer])
results = asyncio.run(evaluation.evaluate(my_pipeline))
```

### Step 5: Verify Traces

After running the instrumented code:

```
Use query_weave_traces_tool with entity_name and project_name to verify traces appeared.
Use metadata_only=True for a quick check.
```

Provide the user with the Weave UI link: `https://wandb.ai/{entity}/{project}/weave/traces`

## Important Rules

1. **Only one `weave.init()`** -- call it once at the entry point, not in every file
2. **Auto-patching is on by default** -- no need for `weave.patch_openai()` unless disabled
3. **`@weave.op()` needs parentheses** -- `@weave.op` without `()` will not work correctly
4. **Environment variable** -- set `WANDB_API_KEY` or run `wandb login` first
5. **Async support** -- `@weave.op()` works on both sync and async functions

## Troubleshooting

- "No traces appearing" -- check `WANDB_API_KEY` is set, project name is correct
- "Module not found" -- run `pip install weave`
- "Traces appear but no LLM calls" -- ensure `weave.init()` is called BEFORE importing the LLM library, or check that implicit patching is not disabled
