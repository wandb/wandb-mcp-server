"""Seed a W&B/Weave project with sample data for skill evaluations.

Creates sample W&B runs (with metrics) and Weave traces (with success/error mix)
so that skill evals can verify MCP tools can access real data.

Profiles:
    default   -- Generic LLM runs and traces.
    hackathon -- Mistral fine-tuning runs with LoRA configs, Mistral agent traces.

Idempotent: checks for existing data before creating. Configurable via env vars.

Usage:
    python -m skills._evals.seed_project                      # default profile
    python -m skills._evals.seed_project --profile hackathon  # hackathon profile
    python -m skills._evals.seed_project --force              # recreate even if exists
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


def seed_hackathon_wandb_runs():
    """Create Mistral fine-tuning runs for the hackathon profile.

    Simulates fine-tuning Mistral models with LoRA, logging loss, eval_accuracy,
    eval_loss, and learning_rate. Includes a LoRA adapter artifact.
    """
    import wandb

    print(f"Seeding {NUM_RUNS} Mistral fine-tuning runs in {EVAL_SEED_ENTITY}/{EVAL_SEED_PROJECT}...")

    configs = [
        {"model": "mistral-small-latest", "lora_rank": 16, "learning_rate": 2e-4, "batch_size": 8, "epochs": 3, "framework": "unsloth"},
        {"model": "mistral-small-latest", "lora_rank": 32, "learning_rate": 1e-4, "batch_size": 16, "epochs": 5, "framework": "unsloth"},
        {"model": "codestral-latest", "lora_rank": 16, "learning_rate": 3e-4, "batch_size": 4, "epochs": 3, "framework": "transformers"},
        {"model": "ministral-8b-latest", "lora_rank": 8, "learning_rate": 5e-5, "batch_size": 32, "epochs": 10, "framework": "art"},
        {"model": "mistral-medium-latest", "lora_rank": 64, "learning_rate": 1e-4, "batch_size": 8, "epochs": 5, "framework": "mistral-api"},
    ]

    for i in range(min(NUM_RUNS, len(configs))):
        cfg = configs[i]
        run = wandb.init(
            entity=EVAL_SEED_ENTITY,
            project=EVAL_SEED_PROJECT,
            name=f"mistral-finetune-{cfg['model'].split('-')[0]}-r{cfg['lora_rank']}",
            config=cfg,
            tags=["eval-seed", "hackathon", "mistral", f"lora-r{cfg['lora_rank']}"],
            reinit=True,
        )

        base_loss = 1.8 - (i * 0.15)
        base_acc = 0.55 + (i * 0.07)
        for step in range(50):
            decay = (step / 50) * 0.85
            noise = random.uniform(-0.03, 0.03)
            run.log({
                "loss": max(0.05, base_loss * (1 - decay) + noise),
                "eval_accuracy": min(0.98, base_acc + decay * 0.35 + noise),
                "eval_loss": max(0.1, base_loss * (1 - decay * 0.6) + noise * 1.5),
                "learning_rate": cfg["learning_rate"] * (1 - step / 60),
            })

        artifact = wandb.Artifact(
            f"lora-adapter-{cfg['model'].split('-')[0]}-r{cfg['lora_rank']}",
            type="model",
            metadata={"base_model": cfg["model"], "lora_rank": cfg["lora_rank"]},
        )
        run.log_artifact(artifact)

        run.finish()
        print(f"  Created run {i+1}/{NUM_RUNS}: {run.name} (artifact: {artifact.name})")


def seed_hackathon_weave_traces():
    """Create Mistral agent traces for the hackathon profile.

    Simulates a Mistral-powered agent pipeline with tool use, producing
    traces that mirror what hackathon participants will generate.
    """
    import weave

    weave.init(f"{EVAL_SEED_ENTITY}/{EVAL_SEED_PROJECT}")
    print(f"Seeding {NUM_TRACES} Mistral agent traces in {EVAL_SEED_ENTITY}/{EVAL_SEED_PROJECT}...")

    @weave.op()
    def mistral_chat_complete(prompt: str, model: str = "mistral-small-latest") -> str:
        """Simulated Mistral API call."""
        time.sleep(random.uniform(0.1, 0.5))
        return f"Mistral response to: {prompt[:40]}..."

    @weave.op()
    def mistral_agent_pipeline(query: str, tools: list[str] | None = None) -> dict:
        """Simulated Mistral agent with tool use."""
        response = mistral_chat_complete(prompt=query)
        return {
            "query": query,
            "response": response,
            "model": "mistral-small-latest",
            "tools_used": tools or [],
            "tokens": {"input": random.randint(100, 800), "output": random.randint(50, 400)},
        }

    @weave.op()
    def mistral_failing_call(prompt: str) -> str:
        """Simulated Mistral call that fails."""
        time.sleep(random.uniform(0.02, 0.1))
        errors = [
            ("RateLimitError", "Rate limit exceeded for mistral-small-latest. Retry after 30s."),
            ("TimeoutError", "Mistral API request timed out after 60s"),
            ("ValidationError", "Input too long: 33,000 tokens exceeds 32,768 limit"),
            ("APIError", "Mistral API returned 503: Service temporarily unavailable"),
        ]
        err_type, err_msg = random.choice(errors)
        raise RuntimeError(f"{err_type}: {err_msg}")

    queries = [
        "Analyze this code for security vulnerabilities",
        "Generate unit tests for my FastAPI endpoint",
        "Explain what this fine-tuning loss curve means",
        "Compare these two model architectures",
        "Create a scoring rubric for my evaluation",
        "What's the best learning rate schedule for LoRA?",
        "Help me design a RAG pipeline with Mistral",
        "Debug my agent's tool use failures",
        "Summarize these evaluation results",
        "What does this error mean in my training run?",
    ]

    success_count = 0
    error_count = 0
    for i in range(NUM_TRACES):
        query = queries[i % len(queries)]
        should_fail = random.random() < ERROR_RATE

        try:
            if should_fail:
                mistral_failing_call(prompt=query)
                error_count += 1
            else:
                tools = random.choice([None, ["web_search"], ["code_exec"], ["web_search", "code_exec"]])
                mistral_agent_pipeline(query=query, tools=tools)
                success_count += 1
        except Exception:
            error_count += 1

    print(f"  Created {NUM_TRACES} traces: {success_count} success, {error_count} error")
    weave.finish()


def check_existing_data(tag: str = "eval-seed") -> bool:
    """Check if seed data already exists."""
    try:
        import wandb

        api = wandb.Api()
        runs = api.runs(f"{EVAL_SEED_ENTITY}/{EVAL_SEED_PROJECT}", filters={"tags": tag})
        run_list = list(runs)
        if len(run_list) >= NUM_RUNS:
            print(f"Seed data already exists: {len(run_list)} runs with tag '{tag}'. Skipping.")
            return True
    except Exception:
        pass
    return False


def main():
    """Seed the eval project with sample data."""
    import argparse

    parser = argparse.ArgumentParser(description="Seed W&B project for skill evals")
    parser.add_argument("--profile", default="default", choices=["default", "hackathon"],
                        help="Seed data profile (default: default)")
    parser.add_argument("--force", action="store_true", help="Recreate even if data exists")
    args = parser.parse_args()

    print("=" * 60)
    print("MCP Skill Eval Seed Project")
    print(f"  Profile: {args.profile}")
    print(f"  Entity:  {EVAL_SEED_ENTITY}")
    print(f"  Project: {EVAL_SEED_PROJECT}")
    print(f"  Runs:    {NUM_RUNS}")
    print(f"  Traces:  {NUM_TRACES}")
    print(f"  Error rate: {ERROR_RATE:.0%}")
    print("=" * 60)

    tag = "hackathon" if args.profile == "hackathon" else "eval-seed"

    if not args.force and check_existing_data(tag):
        print("\nUse --force to recreate seed data.")
        return

    if args.profile == "hackathon":
        seed_hackathon_wandb_runs()
        seed_hackathon_weave_traces()
    else:
        seed_wandb_runs()
        seed_weave_traces()

    print("\nSeed complete. Export these for downstream use:")
    print(f"  export MCP_EVAL_SEED_ENTITY={EVAL_SEED_ENTITY}")
    print(f"  export MCP_EVAL_SEED_PROJECT={EVAL_SEED_PROJECT}")


if __name__ == "__main__":
    main()
