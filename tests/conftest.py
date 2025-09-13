"""
Pytest conftest.py for MCP Server test suite.

This file contains shared fixtures and hooks for the test suite, with a particular
focus on managing Weave evaluation logging in a distributed testing environment
using pytest-xdist.

The Weave aggregation logic has been extracted to `weave_test_aggregator.py` to
keep this file focused on pytest-specific concerns.

Problem with pytest-xdist and session-level Weave Logging:
When using pytest-xdist for parallel test execution, the `pytest_sessionfinish` 
hook runs in each worker process AND the master process. To avoid duplicate Weave
evaluations, we ensure aggregation only happens in the master process.

Solution:
- Worker detection via `session.config.workerinput`
- Weave aggregation only runs when `worker_id == "master"`
- All Weave logic is delegated to the `WeaveTestAggregator` class
"""

import json
import logging
import os
import uuid
from datetime import datetime

import pytest
from dotenv import load_dotenv

from .weave_test_aggregator import aggregate_and_log_test_results

# Load environment variables
load_dotenv()

# Disable Weave tracing in worker processes by default
os.environ["WEAVE_DISABLED"] = "true"
os.environ["WANDB_SILENT"] = "true"

# Configure logging
logger = logging.getLogger("pytest.conftest")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

logger.info(f"Initial WEAVE_DISABLED set to: {os.environ.get('WEAVE_DISABLED')}")

# Weave/W&B configuration
WANDB_TEST_SUITE_PROJECT = os.environ.get("WANDB_PROJECT", "wandb-mcp-server-test-suite-outputs")
WANDB_TEST_SUITE_ENTITY = os.environ.get("WANDB_ENTITY", "wandb-applied-ai-team")
WEAVE_RESULTS_DIR_NAME = "weave_eval_results_json"


@pytest.fixture(scope="session", autouse=True)
def setup_weave_session_config(request):
    """Session-wide setup for Weave configuration."""
    logger.info(f"Pytest session starting. Target Weave project: {WANDB_TEST_SUITE_ENTITY}/{WANDB_TEST_SUITE_PROJECT}")


def pytest_configure(config):
    """Configure pytest settings, particularly for async tests."""
    if hasattr(config.option, "asyncio_mode"):
        config.option.asyncio_mode = "auto"
        config.option.asyncio_default_fixture_loop_scope = "function"


class DateTimeEncoder(json.JSONEncoder):
    """JSON encoder that handles datetime objects."""
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


@pytest.fixture(scope="session")
def weave_results_dir(tmp_path_factory):
    """Create a session-scoped temporary directory for Weave result files."""
    results_dir = tmp_path_factory.mktemp(WEAVE_RESULTS_DIR_NAME, numbered=False)
    logger.info(f"Session temp results directory created: {results_dir}")
    yield results_dir


def pytest_sessionfinish(session):
    """
    Handle session finish - aggregate and log Weave results from master process only.
    
    This hook runs in both worker and master processes when using pytest-xdist.
    We ensure Weave aggregation only happens once by checking the worker_id.
    """
    invocation_id = str(uuid.uuid4())
    
    # Determine if this is a worker or master process
    worker_id = "master"
    workerinput = getattr(session.config, "workerinput", None)
    if workerinput is not None:
        worker_id = workerinput.get("workerid", "worker_unknown")
    
    logger.info(f"pytest_sessionfinish invoked (ID: {invocation_id}, PID: {os.getpid()}, Worker: {worker_id})")
    
    if worker_id != "master":
        logger.info(f"WORKER_LOGIC_SKIP: Skipping aggregation for worker '{worker_id}' (ID: {invocation_id})")
        return
    
    logger.info(f"MASTER_LOGIC_RUN: Running Weave aggregation in master process (ID: {invocation_id})")
    
    # Temporarily enable Weave for the master process
    original_weave_disabled = os.environ.get("WEAVE_DISABLED")
    logger.info(f"(ID: {invocation_id}) Original WEAVE_DISABLED: {original_weave_disabled}")
    
    try:
        os.environ["WEAVE_DISABLED"] = "false"
        logger.info(f"(ID: {invocation_id}) WEAVE_DISABLED temporarily set to 'false' for master")
        
        # Get base temporary directory
        try:
            base_tmp_dir = session.config._tmp_path_factory.getbasetemp()
            logger.info(f"(ID: {invocation_id}) Base temp directory: {base_tmp_dir}")
        except Exception as e:
            logger.error(f"(ID: {invocation_id}) Error accessing temp directory: {e}", exc_info=True)
            return
        
        # Delegate to the aggregator
        success = aggregate_and_log_test_results(
            entity=WANDB_TEST_SUITE_ENTITY,
            project=WANDB_TEST_SUITE_PROJECT,
            base_tmp_dir=base_tmp_dir,
            invocation_id=invocation_id,
            session_config=session.config,
            results_dir_name=WEAVE_RESULTS_DIR_NAME
        )
        
        if success:
            logger.info(f"(ID: {invocation_id}) Weave aggregation completed successfully")
        else:
            logger.warning(f"(ID: {invocation_id}) Weave aggregation completed with issues")
            
    finally:
        # Restore original WEAVE_DISABLED setting
        if original_weave_disabled is None:
            if os.environ.get("WEAVE_DISABLED") == "false":
                del os.environ["WEAVE_DISABLED"]
        else:
            os.environ["WEAVE_DISABLED"] = original_weave_disabled
        
        logger.info(f"(ID: {invocation_id}) WEAVE_DISABLED restored to: {os.environ.get('WEAVE_DISABLED')}")
