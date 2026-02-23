# Framework Detection Patterns

Use these patterns to identify which LLM framework a user's codebase is using.

## File Patterns to Search

| Glob | Suggests |
|------|----------|
| `requirements.txt`, `pyproject.toml`, `setup.py` | Check for `openai`, `anthropic`, `langchain`, `litellm`, `weave` |
| `*.py` containing `import openai` | OpenAI SDK |
| `*.py` containing `from anthropic` | Anthropic SDK |
| `*.py` containing `from langchain` | LangChain |
| `*.py` containing `ChatOpenAI` | LangChain with OpenAI |
| `*.py` containing `ChatAnthropic` | LangChain with Anthropic |
| `*.py` containing `import instructor` | Instructor (structured outputs) |
| `*.py` containing `import litellm` | LiteLLM (multi-provider) |
| `*.py` containing `import weave` | Already using Weave |

## Dependency Check Order

1. Check `pyproject.toml` `[project.dependencies]` or `[tool.poetry.dependencies]`
2. Check `requirements.txt` / `requirements/*.txt`
3. Check `setup.py` `install_requires`
4. Grep Python files for import statements

## Entry Point Detection

Look for the application entry point to place `weave.init()`:

| Pattern | File |
|---------|------|
| `if __name__ == "__main__"` | Script entry point |
| `app = FastAPI()` | FastAPI app |
| `app = Flask(__name__)` | Flask app |
| `def handler(event, context)` | AWS Lambda |
| `@click.command()` | CLI entry point |
| `def main()` | Conventional entry point |
