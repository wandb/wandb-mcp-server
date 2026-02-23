"""Seed a W&B/Weave project with sample data for skill evaluations.

Creates sample W&B runs (with metrics) and Weave traces (with success/error mix)
so that skill evals can verify MCP tools can access real data.

Idempotent: checks for existing data before creating. Configurable via env vars.

Usage:
    python -m skills._evals.seed_project
    # or
    EVAL_SEED_ENTITY=my-team EVAL_SEED_PROJECT=my-project python -m skills._evals.seed_project
"""

import os
import random
import sys
import time

EVAL_SEED_ENTITY = os.environ.get("MCP_EVAL_SEED_ENTITY", os.environ.get("MCP_LOGS_WANDB_ENTITY", "a-sh0ts"))
EVAL_SEED_PROJECT = os.environ.get("MCP_EVAL_SEED_PROJECT", "mcp-skill-eval-seed")
NUM_RUNS = int(os.environ.get("EVAL_SEED_NUM_RUNS", "5"))
NUM_TRACES = int(os.environ.get("EVAL_SEED_NUM_TRACES", "20"))
ERROR_RATE = float(os.environ.get("EVAL_SEED_ERROR_RATE", "0.2"))


def seed_wandb_runs():
    """Create sample W&B runs with training metrics.

    Creates NUM_RUNS runs with loss, accuracy, eval_loss, and learning_rate
    metrics logged over 50 steps each. Each run has a distinct config.
    """
    import wandb

    print(f"Seeding {NUM_RUNS} W&B runs in {EVAL_SEED_ENTITY}/{EVAL_SEED_PROJECT}...")

    configs = [
        {"model": "gpt-4o-mini", "learning_rate": 1e-4, "batch_size": 32, "epochs": 10},
        {"model": "gpt-4o", "learning_rate": 5e-5, "batch_size": 16, "epochs": 20},
        {"model": "claude-sonnet", "learning_rate": 3e-4, "batch_size": 64, "epochs": 5},
        {"model": "llama-3.1-8b", "learning_rate": 2e-4, "batch_size": 32, "epochs": 15},
        {"model": "mistral-7b", "learning_rate": 1e-3, "batch_size": 48, "epochs": 8},
    ]

    for i in range(min(NUM_RUNS, len(configs))):
        cfg = configs[i]
        run = wandb.init(
            entity=EVAL_SEED_ENTITY,
            project=EVAL_SEED_PROJECT,
            name=f"eval-seed-run-{i}",
            config=cfg,
            tags=["eval-seed", f"model-{cfg['model']}"],
            reinit=True,
        )

        base_loss = 2.0 - (i * 0.2)
        base_acc = 0.5 + (i * 0.08)
        for step in range(50):
            decay = (step / 50) * 0.8
            noise = random.uniform(-0.05, 0.05)
            run.log({
                "loss": max(0.1, base_loss * (1 - decay) + noise),
                "accuracy": min(0.99, base_acc + decay * 0.4 + noise),
                "eval_loss": max(0.15, base_loss * (1 - decay * 0.7) + noise * 2),
                "learning_rate": cfg["learning_rate"] * (1 - step / 100),
            })

        run.finish()
        print(f"  Created run {i+1}/{NUM_RUNS}: {run.name}")


def seed_weave_traces():
    """Create sample Weave traces with a mix of success and error statuses.

    Creates NUM_TRACES traces using @weave.op() decorated functions.
    ERROR_RATE fraction of traces will raise exceptions.
    """
    import weave

    weave.init(f"{EVAL_SEED_ENTITY}/{EVAL_SEED_PROJECT}")
    print(f"Seeding {NUM_TRACES} Weave traces in {EVAL_SEED_ENTITY}/{EVAL_SEED_PROJECT}...")

    @weave.op()
    def sample_llm_call(prompt: str, model: str = "gpt-4o-mini") -> str:
        """Simulated LLM call for eval seeding."""
        time.sleep(0.05)
        return f"Response to: {prompt[:50]}... (model={model})"

    @weave.op()
    def sample_pipeline(query: str) -> dict:
        """Simulated pipeline that calls an LLM."""
        result = sample_llm_call(prompt=query)
        return {"query": query, "response": result, "tokens": random.randint(50, 500)}

    @weave.op()
    def sample_failing_call(prompt: str) -> str:
        """Simulated call that raises an error."""
        time.sleep(0.02)
        errors = [
            ("RateLimitError", "Rate limit exceeded. Please retry after 60s."),
            ("ValidationError", "Input validation failed: prompt too long"),
            ("TimeoutError", "Request timed out after 30s"),
            ("APIError", "Internal server error from upstream API"),
        ]
        err_type, err_msg = random.choice(errors)
        raise RuntimeError(f"{err_type}: {err_msg}")

    queries = [
        "What is machine learning?",
        "Explain gradient descent",
        "How does attention work in transformers?",
        "Compare BERT and GPT architectures",
        "What are embedding vectors?",
        "Explain the bias-variance tradeoff",
        "How do you fine-tune a language model?",
        "What is RLHF?",
        "Explain chain-of-thought prompting",
        "What are mixture-of-experts models?",
    ]

    success_count = 0
    error_count = 0
    for i in range(NUM_TRACES):
        query = queries[i % len(queries)]
        should_fail = random.random() < ERROR_RATE

        try:
            if should_fail:
                sample_failing_call(prompt=query)
                error_count += 1
            else:
                sample_pipeline(query=query)
                success_count += 1
        except Exception:
            error_count += 1

    print(f"  Created {NUM_TRACES} traces: {success_count} success, {error_count} error")
    weave.finish()


def check_existing_data() -> bool:
    """Check if seed data already exists."""
    try:
        import wandb

        api = wandb.Api()
        runs = api.runs(f"{EVAL_SEED_ENTITY}/{EVAL_SEED_PROJECT}", filters={"tags": "eval-seed"})
        run_list = list(runs)
        if len(run_list) >= NUM_RUNS:
            print(f"Seed data already exists: {len(run_list)} runs found. Skipping.")
            return True
    except Exception:
        pass
    return False


def main():
    """Seed the eval project with sample data."""
    print("=" * 60)
    print("MCP Skill Eval Seed Project")
    print(f"  Entity:  {EVAL_SEED_ENTITY}")
    print(f"  Project: {EVAL_SEED_PROJECT}")
    print(f"  Runs:    {NUM_RUNS}")
    print(f"  Traces:  {NUM_TRACES}")
    print(f"  Error rate: {ERROR_RATE:.0%}")
    print("=" * 60)

    if "--force" not in sys.argv and check_existing_data():
        print("\nUse --force to recreate seed data.")
        return

    seed_wandb_runs()
    seed_weave_traces()

    print("\nSeed complete. Export these for downstream use:")
    print(f"  export MCP_EVAL_SEED_ENTITY={EVAL_SEED_ENTITY}")
    print(f"  export MCP_EVAL_SEED_PROJECT={EVAL_SEED_PROJECT}")


if __name__ == "__main__":
    main()
