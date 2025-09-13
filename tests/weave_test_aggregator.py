"""
Weave Test Result Aggregation Utility

This module handles the aggregation and logging of test results to Weave/W&B
for the MCP Server test suite. It's designed to work with pytest-xdist by
ensuring that Weave evaluation logging happens only once across multiple
worker processes.

The main entry point is `aggregate_and_log_test_results()` which should be
called from the master pytest process after all tests have completed.
"""

import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Try to import Weave dependencies
try:
    import weave
    from weave import EvaluationLogger
    from weave.trace.context.weave_client_context import WeaveInitError
    WEAVE_AVAILABLE = True
except ImportError:
    weave = None
    EvaluationLogger = None
    WeaveInitError = Exception
    WEAVE_AVAILABLE = False
    logger.warning("Weave SDK not available. Weave evaluation logging will be skipped.")


class WeaveTestAggregator:
    """Handles aggregation and logging of test results to Weave."""
    
    def __init__(self, entity: str, project: str, results_dir_name: str = "weave_eval_results_json"):
        self.entity = entity
        self.project = project
        self.results_dir_name = results_dir_name
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def _initialize_weave(self, invocation_id: str) -> bool:
        """Initialize Weave connection. Returns True if successful."""
        if not WEAVE_AVAILABLE:
            self.logger.warning(f"(ID: {invocation_id}) Weave SDK not available.")
            return False
        
        if not self.entity or not self.project:
            self.logger.warning(f"(ID: {invocation_id}) Entity or project not set.")
            return False
        
        try:
            self.logger.info(f"(ID: {invocation_id}) Initializing Weave: {self.entity}/{self.project}")
            weave.init(f"{self.entity}/{self.project}")
            self.logger.info(f"(ID: {invocation_id}) Weave initialized successfully.")
            return True
        except WeaveInitError as e:
            self.logger.error(f"(ID: {invocation_id}) WeaveInitError: {e}", exc_info=True)
            return False
        except Exception as e:
            self.logger.error(f"(ID: {invocation_id}) Error initializing Weave: {e}", exc_info=True)
            return False
    
    def _discover_json_files(self, base_tmp_dir: Path, invocation_id: str) -> List[Path]:
        """Discover JSON result files in the temporary directory structure."""
        json_files = []
        
        try:
            self.logger.info(f"(ID: {invocation_id}) Searching base directory: {base_tmp_dir}")
            
            for item in base_tmp_dir.iterdir():
                if item.is_dir():
                    # Check for results directory within subdirectories (xdist workers)
                    target_results_dir = item / self.results_dir_name
                    if target_results_dir.is_dir():
                        self.logger.info(f"(ID: {invocation_id}) Found results dir: {target_results_dir}")
                        json_files.extend(list(target_results_dir.glob("*.json")))
                    
                    # Check if the item itself is the results directory (non-xdist)
                    elif item.name == self.results_dir_name and item.is_dir():
                        self.logger.info(f"(ID: {invocation_id}) Found non-xdist results dir: {item}")
                        json_files.extend(list(item.glob("*.json")))
            
            # Deduplicate and sort
            json_files = sorted(list(set(json_files)))
            self.logger.info(f"(ID: {invocation_id}) Found {len(json_files)} JSON files total.")
            
        except Exception as e:
            self.logger.error(f"(ID: {invocation_id}) Error discovering JSON files: {e}", exc_info=True)
        
        return json_files
    
    def _load_test_data(self, json_files: List[Path], invocation_id: str) -> List[Dict]:
        """Load and parse test data from JSON files."""
        all_test_data = []
        
        for json_file in json_files:
            try:
                with open(json_file, "r") as f:
                    data = json.load(f)
                    all_test_data.append(data)
            except Exception as e:
                self.logger.error(f"(ID: {invocation_id}) Error reading {json_file}: {e}", exc_info=True)
        
        return all_test_data
    
    def _group_test_data_by_source(self, test_data: List[Dict]) -> Dict[str, List[Dict]]:
        """Group test data by source test file name."""
        grouped_data = defaultdict(list)
        
        for item in test_data:
            source_file = item.get("metadata", {}).get("source_test_file_name", "unknown_source_file")
            grouped_data[source_file].append(item)
        
        return dict(grouped_data)
    
    def _create_eval_names(self, source_file: str, metadata: Dict) -> Tuple[str, str]:
        """Create evaluation and dataset names from source file and metadata."""
        git_commit = metadata.get("git_commit_id", "unknown_git_commit")
        sanitized_source = source_file.replace("_", "-")
        
        eval_name = f"mcp-eval_{sanitized_source}_{git_commit}"
        dataset_name = f"{sanitized_source}_tests"
        
        return eval_name, dataset_name
    
    def _log_test_group_to_weave(self, source_file: str, test_data: List[Dict], invocation_id: str) -> bool:
        """Log a group of tests from the same source file to Weave."""
        if not test_data:
            self.logger.warning(f"(ID: {invocation_id}) No test data for '{source_file}'")
            return False
        
        # Create evaluation logger
        first_metadata = test_data[0].get("metadata", {})
        eval_name, dataset_name = self._create_eval_names(source_file, first_metadata)
        git_commit = first_metadata.get("git_commit_id", "unknown_git_commit")
        
        self.logger.info(f"(ID: {invocation_id}) Logging {len(test_data)} tests from '{source_file}' as '{eval_name}'")
        
        try:
            eval_logger = EvaluationLogger(
                name=eval_name,
                model=git_commit,
                dataset=dataset_name,
            )
        except Exception as e:
            self.logger.error(f"(ID: {invocation_id}) Failed to create EvaluationLogger for '{source_file}': {e}", exc_info=True)
            return False
        
        # Log individual test predictions
        total_logged = 0
        passed_logged = 0
        all_latencies = []
        
        for test_item in test_data:
            try:
                metadata = test_item.get("metadata", {})
                inputs = dict(test_item.get("inputs", {}))
                output = test_item.get("output", {})
                score_value = test_item.get("score", False)
                metrics = test_item.get("metrics", {})
                
                # Enrich inputs with metadata
                if "test_case_index" in metadata:
                    inputs["_test_case_index"] = metadata["test_case_index"]
                if "sample_name" in metadata:
                    inputs["_sample_name"] = metadata["sample_name"]
                inputs["_source_test_file_name"] = metadata.get("source_test_file_name", source_file)
                inputs["_original_test_query_text"] = metadata.get("test_query_text", "N/A")
                
                # Log prediction
                score_logger = eval_logger.log_prediction(inputs=inputs, output=output)
                score_logger.log_score(scorer="test_passed", score=bool(score_value))
                
                # Log execution latency if available
                execution_latency = metrics.get("execution_latency_seconds")
                if execution_latency is not None:
                    score_logger.log_score(scorer="execution_latency_seconds", score=float(execution_latency))
                    all_latencies.append(float(execution_latency))
                
                score_logger.finish()
                total_logged += 1
                if score_value:
                    passed_logged += 1
                    
            except Exception as e:
                test_id = metadata.get("test_case_index", metadata.get("sample_name", "unknown"))
                self.logger.error(f"(ID: {invocation_id}) Error logging test '{test_id}': {e}", exc_info=True)
        
        # Log summary metrics
        if total_logged > 0:
            summary_metrics = {
                "count_tests_logged": total_logged,
                "pass_rate": passed_logged / total_logged if total_logged else 0,
            }
            
            if all_latencies:
                summary_metrics.update({
                    "avg_execution_latency_s": sum(all_latencies) / len(all_latencies),
                    "min_execution_latency_s": min(all_latencies),
                    "max_execution_latency_s": max(all_latencies),
                    "total_execution_latency_s": sum(all_latencies),
                })
            
            try:
                eval_logger.log_summary(summary_metrics)
                self.logger.info(f"(ID: {invocation_id}) Successfully logged summary for '{eval_name}': {summary_metrics}")
                return True
            except Exception as e:
                self.logger.error(f"(ID: {invocation_id}) Failed to log summary for '{eval_name}': {e}", exc_info=True)
        else:
            self.logger.info(f"(ID: {invocation_id}) No tests logged for '{eval_name}'")
        
        return False
    
    def aggregate_and_log_results(self, base_tmp_dir: Path, invocation_id: str, 
                                 session_config: Optional[object] = None) -> bool:
        """
        Main entry point for aggregating and logging test results to Weave.
        
        Args:
            base_tmp_dir: Base temporary directory containing test result files
            invocation_id: Unique identifier for this aggregation run
            session_config: Optional pytest session config for additional metadata
            
        Returns:
            True if aggregation was successful, False otherwise
        """
        self.logger.info(f"(ID: {invocation_id}) Starting Weave test result aggregation")
        
        # Initialize Weave
        if not self._initialize_weave(invocation_id):
            return False
        
        # Discover JSON result files
        json_files = self._discover_json_files(base_tmp_dir, invocation_id)
        if not json_files:
            self.logger.info(f"(ID: {invocation_id}) No JSON result files found")
            return False
        
        # Load test data
        all_test_data = self._load_test_data(json_files, invocation_id)
        if not all_test_data:
            self.logger.info(f"(ID: {invocation_id}) No valid test data loaded")
            return False
        
        # Group by source file and log to Weave
        grouped_data = self._group_test_data_by_source(all_test_data)
        self.logger.info(f"(ID: {invocation_id}) Processing {len(grouped_data)} test file groups")
        
        success_count = 0
        for source_file, file_test_data in grouped_data.items():
            if self._log_test_group_to_weave(source_file, file_test_data, invocation_id):
                success_count += 1
        
        self.logger.info(f"(ID: {invocation_id}) Successfully logged {success_count}/{len(grouped_data)} test groups")
        return success_count > 0


def aggregate_and_log_test_results(entity: str, project: str, base_tmp_dir: Path, 
                                  invocation_id: str, session_config: Optional[object] = None,
                                  results_dir_name: str = "weave_eval_results_json") -> bool:
    """
    Convenience function for aggregating and logging test results to Weave.
    
    This is the main entry point that should be called from pytest hooks.
    """
    aggregator = WeaveTestAggregator(entity, project, results_dir_name)
    return aggregator.aggregate_and_log_results(base_tmp_dir, invocation_id, session_config)