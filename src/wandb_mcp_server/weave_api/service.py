"""
Service layer for Weave API.

This module provides high-level services for querying and processing Weave traces.
It orchestrates the client, query builder, and processor components.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

from wandb_mcp_server.utils import get_rich_logger, get_server_args
from wandb_mcp_server.config import WF_TRACE_SERVER_URL
from wandb_mcp_server.weave_api.client import WeaveApiClient
from wandb_mcp_server.api_client import WandBApiManager
from wandb_mcp_server.weave_api.models import QueryResult
from wandb_mcp_server.weave_api.processors import TraceProcessor
from wandb_mcp_server.weave_api.query_builder import QueryBuilder

# Import CallSchema to validate column names
try:
    from weave.trace_server.trace_server_interface import CallSchema

    VALID_COLUMNS = set(CallSchema.__annotations__.keys())
    HAVE_CALL_SCHEMA = True
except ImportError:
    # Fallback if CallSchema isn't available
    VALID_COLUMNS = {
        "id",
        "project_id",
        "op_name",
        "display_name",
        "trace_id",
        "parent_id",
        "started_at",
        "attributes",
        "inputs",
        "ended_at",
        "exception",
        "output",
        "summary",
        "wb_user_id",
        "wb_run_id",
        "deleted_at",
        "storage_size_bytes",
        "total_storage_size_bytes",
    }
    HAVE_CALL_SCHEMA = False

logger = get_rich_logger(__name__)


class TraceService:
    """Service for querying and processing Weave traces."""

    # Define cost fields once as a class constant
    COST_FIELDS = {"total_cost", "completion_cost", "prompt_cost"}

    # Define synthetic columns that shouldn't be passed to the API but can be reconstructed
    SYNTHETIC_COLUMNS = {"costs"}

    # Define latency field mapping
    LATENCY_FIELD_MAPPING = {"latency_ms": "summary.weave.latency_ms"}

    def __init__(
        self,
        api_key: Optional[str] = None,
        server_url: Optional[str] = None,
        retries: int = 3,
        timeout: int = 10,
    ):
        """Initialize the TraceService.

        Args:
            api_key: W&B API key. If not provided, uses WANDB_API_KEY env var.
            server_url: Weave API server URL. Defaults to env-driven config.
            retries: Number of retries for failed requests.
            timeout: Request timeout in seconds.
        """
        # If no API key provided, try to get from context only (no fallbacks!)
        if api_key is None:
            # Get from context (set by auth middleware for HTTP, at startup for STDIO)
            api_key = WandBApiManager.get_api_key()
            
            # NO FALLBACKS to get_server_args or environment!
            # API key must be explicitly set in context
        
        # Validate we have an API key before creating client
        if not api_key:
            raise ValueError(
                "No W&B API key available in TraceService. "
                "For HTTP: Ensure authentication middleware is configured. "
                "For STDIO: Ensure API key is set at server startup."
            )

        # Pass the resolved API key to WeaveApiClient
        self.client = WeaveApiClient(
            api_key=api_key,
            server_url=server_url or WF_TRACE_SERVER_URL,
            retries=retries,
            timeout=timeout,
        )

        # Initialize collection for invalid columns (for warning messages)
        self.invalid_columns = set()

    def _validate_and_filter_columns(
        self, columns: Optional[List[str]]
    ) -> tuple[Optional[List[str]], List[str], Set[str]]:
        """Validate columns against CallSchema and filter out synthetic/invalid columns.

        Handles mapping of 'latency_ms' to 'summary.weave.latency_ms'.

        Args:
            columns: List of columns.

        Returns:
            Tuple of (filtered_columns_for_api, requested_synthetic_columns, invalid_columns_reported)
        """
        if not columns:
            return (
                None,
                [],
                set(),
            )  # Return None for filtered_columns_for_api if input is None

        filtered_columns_for_api: list[str] = []
        requested_synthetic_columns: list[str] = []
        invalid_columns_reported: set[str] = set()

        processed_columns = (
            set()
        )  # To avoid duplicate processing if a column is listed multiple times

        for col_name in columns:
            if col_name in processed_columns:
                continue
            processed_columns.add(col_name)

            if col_name == "latency_ms":
                # 'latency_ms' is synthetic, its data comes from 'summary.weave.latency_ms'
                requested_synthetic_columns.append("latency_ms")
                # Ensure the source field is requested from the API
                source_field = self.LATENCY_FIELD_MAPPING["latency_ms"]
                if source_field not in filtered_columns_for_api:
                    filtered_columns_for_api.append(source_field)
                # Also ensure 'summary' itself is added if not already, as 'summary.weave.latency_ms' implies 'summary'
                if (
                    "summary" not in filtered_columns_for_api
                    and source_field.startswith("summary.")
                ):
                    filtered_columns_for_api.append("summary")
                logger.info(
                    f"Column 'latency_ms' requested: will be synthesized from '{source_field}'. Added '{source_field}' to API columns."
                )

            elif col_name == "costs":
                # 'costs' is synthetic, its data comes from 'summary.weave.costs'
                requested_synthetic_columns.append("costs")
                # Ensure the source field ('summary') is requested
                if "summary" not in filtered_columns_for_api:
                    filtered_columns_for_api.append("summary")
                logger.info(
                    "Column 'costs' requested: will be synthesized from 'summary.weave.costs'. Added 'summary' to API columns."
                )

            elif col_name == "status":
                # 'status' can be top-level or from 'summary.weave.status'
                requested_synthetic_columns.append("status")
                # Add 'status' to API columns to try fetching top-level first.
                # If not present, it will be synthesized from summary.
                if "status" not in filtered_columns_for_api:
                    filtered_columns_for_api.append("status")
                if (
                    "summary" not in filtered_columns_for_api
                ):  # Also ensure summary for fallback
                    filtered_columns_for_api.append("summary")
                logger.info(
                    "Column 'status' requested: will attempt direct fetch or synthesize from 'summary.weave.status'."
                )

            elif col_name in VALID_COLUMNS:
                # Direct valid column
                if col_name not in filtered_columns_for_api:
                    filtered_columns_for_api.append(col_name)

            elif "." in col_name:  # Potentially a dot-separated path
                base_field = col_name.split(".")[0]
                if base_field in VALID_COLUMNS:
                    # Valid nested field (e.g., "summary.weave.latency_ms", "attributes.foo")
                    if col_name not in filtered_columns_for_api:
                        filtered_columns_for_api.append(col_name)
                    logger.info(
                        f"Nested column field '{col_name}' requested, added to API columns."
                    )
                else:
                    logger.warning(
                        f"Invalid base field '{base_field}' in nested column '{col_name}'. It will be ignored."
                    )
                    invalid_columns_reported.add(col_name)
            else:
                # Neither a direct valid column, nor a recognized synthetic, nor a valid-looking nested path
                logger.warning(
                    f"Invalid column '{col_name}' requested. It will be ignored."
                )
                invalid_columns_reported.add(col_name)

        # Ensure filtered_columns_for_api does not have duplicates and maintains order as much as possible
        # (though order to the API might not matter as much as presence)
        final_filtered_columns_for_api = []
        seen_in_final = set()
        for fc in filtered_columns_for_api:
            if fc not in seen_in_final:
                final_filtered_columns_for_api.append(fc)
                seen_in_final.add(fc)

        return (
            final_filtered_columns_for_api,
            requested_synthetic_columns,
            invalid_columns_reported,
        )

    def _ensure_required_columns_for_synthetic(
        self,
        filtered_columns: Optional[List[str]],
        requested_synthetic_columns: List[str],
    ) -> Optional[List[str]]:
        """Ensure required columns for synthetic fields are included.

        Args:
            filtered_columns: List of columns after filtering out synthetic ones.
            requested_synthetic_columns: List of requested synthetic columns.

        Returns:
            Updated filtered columns list with required columns added.
        """
        if not filtered_columns:
            filtered_columns = []

        required_columns = set(filtered_columns)

        # Add required columns for synthesizing costs
        if "costs" in requested_synthetic_columns:
            # Costs data comes from summary.weave.costs
            if "summary" not in required_columns:
                logger.info("Adding 'summary' column as it's required for costs data")
                required_columns.add("summary")

        # Add other required columns for other synthetic fields as needed

        return list(required_columns)

    def _add_synthetic_columns(
        self,
        traces: List[Dict[str, Any]],
        requested_synthetic_columns: List[str],
        invalid_columns: Set[str],
    ) -> List[Dict[str, Any]]:
        """Add synthetic columns back to the traces and add warnings for invalid columns.

        Args:
            traces: List of trace dictionaries.
            requested_synthetic_columns: List of requested synthetic columns.
            invalid_columns: Set of invalid column names that were requested.

        Returns:
            Updated traces with synthetic columns added and invalid column warnings.
        """
        if not requested_synthetic_columns and not invalid_columns:
            return traces

        updated_traces = []

        for trace in traces:
            updated_trace = trace.copy()

            # Add costs data if requested
            if "costs" in requested_synthetic_columns:
                costs_data = trace.get("summary", {}).get("weave", {}).get("costs", {})
                if costs_data:
                    logger.debug(
                        f"Adding synthetic 'costs' column with {len(costs_data)} providers"
                    )
                    updated_trace["costs"] = costs_data
                else:
                    logger.warning(f"No costs data found in trace {trace.get('id')}")
                    updated_trace["costs"] = {}

            # Add status from summary if requested
            if "status" in requested_synthetic_columns:
                status = trace.get("status")  # Check if it's already in the trace
                if not status:
                    # Extract from summary.weave.status
                    status = trace.get("summary", {}).get("weave", {}).get("status")
                    if status:
                        logger.debug(
                            f"Adding synthetic 'status' from summary: {status}"
                        )
                        updated_trace["status"] = status
                    else:
                        logger.warning(
                            f"No status data found in trace {trace.get('id')}"
                        )
                        updated_trace["status"] = None

            # Add latency_ms from summary if requested
            if "latency_ms" in requested_synthetic_columns:
                latency = trace.get("latency_ms")  # Check if it's already in the trace
                if latency is None:
                    # Extract from summary.weave.latency_ms
                    latency = (
                        trace.get("summary", {}).get("weave", {}).get("latency_ms")
                    )
                    if latency is not None:
                        logger.debug(
                            f"Adding synthetic 'latency_ms' from summary: {latency}"
                        )
                        updated_trace["latency_ms"] = latency
                    else:
                        logger.warning(
                            f"No latency_ms data found in trace {trace.get('id')}"
                        )
                        updated_trace["latency_ms"] = None

            # Add warnings for invalid columns
            for col in invalid_columns:
                warning_message = f"{col} is not a valid column name, no data returned"
                updated_trace[col] = warning_message

            updated_traces.append(updated_trace)

        return updated_traces

    def query_traces(
        self,
        entity_name: str,
        project_name: str,
        filters: Optional[Dict[str, Any]] = None,
        sort_by: str = "started_at",
        sort_direction: str = "desc",
        limit: Optional[int] = None,
        offset: int = 0,
        include_costs: bool = True,
        include_feedback: bool = True,
        columns: Optional[List[str]] = None,
        expand_columns: Optional[List[str]] = None,
        truncate_length: Optional[int] = 200,
        return_full_data: bool = False,
        metadata_only: bool = False,
    ) -> QueryResult:
        """Query traces from the Weave API.

        Args:
            entity_name: Weights & Biases entity name.
            project_name: Weights & Biands project name.
            filters: Dictionary of filter conditions.
            sort_by: Field to sort by.
            sort_direction: Sort direction ('asc' or 'desc').
            limit: Maximum number of results to return.
            offset: Number of results to skip (for pagination).
            include_costs: Include tracked API cost information in the results.
            include_feedback: Include Weave annotations in the results.
            columns: List of specific columns to include in the results.
            expand_columns: List of columns to expand in the results.
            truncate_length: Maximum length for string values.
            return_full_data: Whether to include full untruncated trace data.
            metadata_only: Whether to only include metadata without traces.

        Returns:
            QueryResult object with metadata and optionally traces.
        """
        # Clear invalid columns from previous requests
        self.invalid_columns = set()

        # Special handling for cost-based sorting
        client_side_cost_sort = sort_by in self.COST_FIELDS

        # Handle latency field mapping
        if sort_by in self.LATENCY_FIELD_MAPPING:
            logger.info(
                f"Mapping sort field '{sort_by}' to '{self.LATENCY_FIELD_MAPPING[sort_by]}'"
            )
            server_sort_by = self.LATENCY_FIELD_MAPPING[sort_by]
            server_sort_direction = sort_direction
        elif client_side_cost_sort:
            include_costs = True
            server_sort_by = "started_at"
            server_sort_direction = sort_direction
        elif sort_by == "latency_ms":  # Added specific handling for latency_ms sort
            logger.info(
                f"Sort by 'latency_ms' requested. Will sort by server field '{self.LATENCY_FIELD_MAPPING['latency_ms']}'."
            )
            server_sort_by = self.LATENCY_FIELD_MAPPING["latency_ms"]
            server_sort_direction = sort_direction
        elif "." in sort_by:  # Handles general dot-separated paths
            base_field = sort_by.split(".")[0]
            if base_field in VALID_COLUMNS:
                logger.info(f"Using nested sort field for server: {sort_by}")
                server_sort_by = sort_by
                server_sort_direction = sort_direction
            else:
                logger.warning(
                    f"Invalid base field '{base_field}' in sort_by '{sort_by}', falling back to 'started_at'."
                )
                server_sort_by = "started_at"
                server_sort_direction = sort_direction
        elif sort_by not in VALID_COLUMNS:
            logger.warning(
                f"Invalid sort field '{sort_by}', falling back to 'started_at'."
            )
            server_sort_by = "started_at"
            server_sort_direction = sort_direction
        else:  # sort_by is in VALID_COLUMNS and not a special case
            server_sort_by = sort_by
            server_sort_direction = sort_direction

        # Validate and filter columns using CallSchema
        filtered_api_columns, rs_columns, inv_columns = (
            self._validate_and_filter_columns(columns)
        )

        # Store invalid columns for later
        self.invalid_columns = inv_columns  # Corrected variable name

        # If costs was requested as a column (now checked via rs_columns), make sure to include it
        if "costs" in rs_columns:  # Corrected check
            include_costs = True

        # Manually add latency_ms to synthetic fields if requested - This is now handled in _validate_and_filter_columns
        # if columns and "latency_ms" in columns and "latency_ms" not in requested_synthetic_columns:
        #     requested_synthetic_columns.append("latency_ms")

        # Ensure required columns for synthetic fields are included - This is also largely handled by _validate_and_filter_columns logic
        # filtered_api_columns = self._ensure_required_columns_for_synthetic(filtered_api_columns, rs_columns)

        # Prepare query parameters
        query_params = {
            "entity_name": entity_name,
            "project_name": project_name,
            "filters": filters or {},
            "sort_by": server_sort_by,
            "sort_direction": server_sort_direction,
            "limit": None
            if client_side_cost_sort
            else limit,  # No limit if we're sorting by cost
            "offset": offset,
            "include_costs": include_costs,
            "include_feedback": include_feedback,
            "columns": filtered_api_columns,  # Use the columns intended for the API
            "expand_columns": expand_columns,
        }

        # Build request body
        request_body = QueryBuilder.prepare_query_params(query_params)

        # Extract synthetic fields if any were specified
        synthetic_fields = (
            request_body.pop("_synthetic_fields", [])
            if "_synthetic_fields" in request_body
            else []
        )

        # Make sure all requested synthetic columns are included in synthetic_fields
        for col in rs_columns:  # Use rs_columns
            if col not in synthetic_fields:
                synthetic_fields.append(col)

        # Execute query
        all_traces = list(self.client.query_traces(request_body))

        # Add synthetic columns and invalid column warnings back to the results
        if rs_columns or inv_columns:  # Use corrected variables
            all_traces = self._add_synthetic_columns(
                all_traces, rs_columns, inv_columns
            )

        # Client-side cost-based sorting if needed
        if client_side_cost_sort and all_traces:
            logger.info(f"Performing client-side sorting by {sort_by}")
            # Sort traces by cost
            all_traces.sort(
                key=lambda t: TraceProcessor.get_cost(t, sort_by),
                reverse=(sort_direction == "desc"),
            )
            # Apply limit if specified
            if limit is not None:
                all_traces = all_traces[:limit]

        # If we need to synthesize fields, do it
        if synthetic_fields:
            logger.info(f"Synthesizing fields: {synthetic_fields}")
            all_traces = [
                TraceProcessor.synthesize_fields(trace, synthetic_fields)
                for trace in all_traces
            ]

        # Process traces
        result = TraceProcessor.process_traces(
            traces=all_traces,
            truncate_length=truncate_length or 0,
            return_full_data=return_full_data,
            metadata_only=metadata_only,
        )

        return result

    def query_paginated_traces(
        self,
        entity_name: str,
        project_name: str,
        chunk_size: int = 20,
        filters: Optional[Dict[str, Any]] = None,
        sort_by: str = "started_at",
        sort_direction: str = "desc",
        target_limit: Optional[int] = None,
        include_costs: bool = True,
        include_feedback: bool = True,
        columns: Optional[List[str]] = None,
        expand_columns: Optional[List[str]] = None,
        truncate_length: Optional[int] = 200,
        return_full_data: bool = False,
        metadata_only: bool = False,
    ) -> QueryResult:
        """Query traces with pagination.

        Args:
            entity_name: Weights & Biases entity name.
            project_name: Weights & Biands project name.
            chunk_size: Number of traces to retrieve in each chunk.
            filters: Dictionary of filter conditions.
            sort_by: Field to sort by.
            sort_direction: Sort direction ('asc' or 'desc').
            target_limit: Maximum total number of results to return.
            include_costs: Include tracked API cost information in the results.
            include_feedback: Include Weave annotations in the results.
            columns: List of specific columns to include in the results.
            expand_columns: List of columns to expand in the results.
            truncate_length: Maximum length for string values.
            return_full_data: Whether to include full untruncated trace data.
            metadata_only: Whether to only include metadata without traces.

        Returns:
            QueryResult object with metadata and optionally traces.
        """
        # Special handling for cost-based sorting
        client_side_cost_sort = sort_by in self.COST_FIELDS

        # Determine effective_sort_by for the server
        effective_sort_by = "started_at"  # Default
        if sort_by == "latency_ms":
            effective_sort_by = self.LATENCY_FIELD_MAPPING["latency_ms"]
            logger.info(
                f"Paginated sort by 'latency_ms', server will use '{effective_sort_by}'."
            )
        elif "." in sort_by:
            base_field = sort_by.split(".")[0]
            if base_field in VALID_COLUMNS:
                effective_sort_by = sort_by
                logger.info(
                    f"Paginated sort by nested field '{sort_by}', server will use it directly."
                )
            else:
                logger.warning(
                    f"Paginated sort by invalid nested field '{sort_by}', defaulting to 'started_at'."
                )
        elif (
            sort_by in VALID_COLUMNS and sort_by not in self.COST_FIELDS
        ):  # Exclude COST_FIELDS as they are client-sorted
            effective_sort_by = sort_by
        elif (
            sort_by not in self.COST_FIELDS
        ):  # If not valid and not cost, warn and default
            logger.warning(
                f"Paginated sort by invalid field '{sort_by}', defaulting to 'started_at'."
            )

        # Validate and filter columns using CallSchema
        # Pass the original 'columns'
        filtered_api_columns, rs_columns, inv_columns = (
            self._validate_and_filter_columns(columns)
        )

        # Store invalid columns for later
        self.invalid_columns = inv_columns  # Corrected

        # If costs was requested as a column, make sure to include it
        if "costs" in rs_columns:  # Corrected
            include_costs = True

        # Ensure required columns for synthetic fields are included - Handled by _validate_and_filter_columns
        # filtered_api_columns = self._ensure_required_columns_for_synthetic(filtered_api_columns, rs_columns)

        if client_side_cost_sort:
            logger.info(f"Cost-based sorting detected: {sort_by}")
            all_traces = self._query_for_cost_sorting(
                entity_name=entity_name,
                project_name=project_name,
                filters=filters,
                sort_by=sort_by,
                sort_direction=sort_direction,
                target_limit=target_limit,
                columns=filtered_api_columns,  # Pass filtered columns for API
                expand_columns=expand_columns,
                include_costs=True,
                include_feedback=include_feedback,
                requested_synthetic_columns=rs_columns,  # Pass synthetic columns request
                invalid_columns=inv_columns,  # Pass invalid columns
            )
        else:
            # Normal paginated query logic
            all_traces = []
            current_offset = 0

            while True:
                logger.info(
                    f"Querying chunk with offset {current_offset}, size {chunk_size}"
                )
                remaining = (
                    target_limit - len(all_traces) if target_limit else chunk_size
                )
                current_chunk_size = (
                    min(chunk_size, remaining) if target_limit else chunk_size
                )

                chunk_result = self.query_traces(
                    entity_name=entity_name,
                    project_name=project_name,
                    filters=filters,
                    sort_by=effective_sort_by,
                    sort_direction=sort_direction,
                    limit=current_chunk_size,
                    offset=current_offset,
                    include_costs=include_costs,
                    include_feedback=include_feedback,
                    columns=columns,  # Pass original 'columns' here, query_traces will validate and filter.
                    # This ensures that if 'latency_ms' was requested, it's handled correctly
                    # by the nested call to _validate_and_filter_columns inside query_traces.
                    expand_columns=expand_columns,
                    return_full_data=True,  # We want raw data for now
                    metadata_only=False,
                )

                # Get the traces from the QueryResult and handle both None and empty list cases
                traces_from_chunk = (
                    chunk_result.traces if chunk_result and chunk_result.traces else []
                )
                if not traces_from_chunk:
                    break

                all_traces.extend(traces_from_chunk)

                if len(traces_from_chunk) < current_chunk_size or (
                    target_limit and len(all_traces) >= target_limit
                ):
                    break

                current_offset += chunk_size

        # Process all traces at once with appropriate parameters
        if target_limit and all_traces:
            all_traces = all_traces[:target_limit]

        result = TraceProcessor.process_traces(
            traces=all_traces,
            truncate_length=truncate_length or 0,
            return_full_data=return_full_data,
            metadata_only=metadata_only,
        )
        logger.debug(
            f"Final result from query_paginated_traces:\n\n{len(result.model_dump_json(indent=2))}\n"
        )
        assert isinstance(result, QueryResult), (
            f"Result type must be a QueryResult, found: {type(result)}"
        )
        return result

    def _query_for_cost_sorting(
        self,
        entity_name: str,
        project_name: str,
        filters: Optional[Dict[str, Any]] = None,
        sort_by: str = "total_cost",
        sort_direction: str = "desc",
        target_limit: Optional[int] = None,
        columns: Optional[List[str]] = None,
        expand_columns: Optional[List[str]] = None,
        include_costs: bool = True,
        include_feedback: bool = True,
        requested_synthetic_columns: Optional[List[str]] = None,
        invalid_columns: Optional[Set[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Special two-stage query logic for cost-based sorting.

        Args:
            entity_name: Weights & Biases entity name.
            project_name: Weights & Biands project name.
            filters: Dictionary of filter conditions.
            sort_by: Cost field to sort by.
            sort_direction: Sort direction ('asc' or 'desc').
            target_limit: Maximum number of results to return.
            columns: List of specific columns to include in the results.
            expand_columns: List of columns to expand in the results.
            include_costs: Include tracked API cost information in the results.
            include_feedback: Include Weave annotations in the results.
            requested_synthetic_columns: List of synthetic columns requested by the user.
            invalid_columns: Set of invalid column names that were requested.

        Returns:
            List of trace dictionaries sorted by the specified cost field.
        """
        if invalid_columns is None:
            invalid_columns = set()

        # First pass: Fetch all trace IDs and costs
        first_pass_query = {
            "entity_name": entity_name,
            "project_name": project_name,
            "filters": filters or {},
            "sort_by": "started_at",  # Use a standard sort for the first pass
            "sort_direction": "desc",
            "limit": 1000000,  # Explicitly set a large limit to get all traces
            "include_costs": True,  # We need costs for sorting
            "include_feedback": False,  # Don't need feedback for the first pass
            "columns": ["id", "summary"],  # Need summary for costs data
        }

        first_pass_request = QueryBuilder.prepare_query_params(first_pass_query)
        first_pass_results = list(self.client.query_traces(first_pass_request))

        logger.info(
            f"First pass of cost sorting request retrieved {len(first_pass_results)} traces"
        )

        # Filter and sort by cost
        filtered_results = [
            t
            for t in first_pass_results
            if TraceProcessor.get_cost(t, sort_by) is not None
        ]

        filtered_results.sort(
            key=lambda t: TraceProcessor.get_cost(t, sort_by),
            reverse=(sort_direction == "desc"),
        )

        # Get the IDs of the top N traces
        top_ids = (
            [t["id"] for t in filtered_results[:target_limit] if "id" in t]
            if target_limit
            else [t["id"] for t in filtered_results if "id" in t]
        )

        logger.info(f"After sorting by {sort_by}, selected {len(top_ids)} trace IDs")

        if not top_ids:
            return []

        # Second pass: Fetch the full details for the selected traces
        second_pass_query = {
            "entity_name": entity_name,
            "project_name": project_name,
            "filters": {"call_ids": top_ids},
            "include_costs": include_costs,
            "include_feedback": include_feedback,
            "columns": columns,
            "expand_columns": expand_columns,
        }

        # Make sure we request summary if costs were requested
        if requested_synthetic_columns and "costs" in requested_synthetic_columns:
            if not columns or "summary" not in columns:
                if not second_pass_query["columns"]:
                    second_pass_query["columns"] = ["summary"]
                elif "summary" not in second_pass_query["columns"]:
                    second_pass_query["columns"].append("summary")
                logger.info("Added 'summary' to columns for cost data retrieval")

        second_pass_request = QueryBuilder.prepare_query_params(second_pass_query)
        second_pass_results = list(self.client.query_traces(second_pass_request))

        logger.info(f"Second pass retrieved {len(second_pass_results)} traces")

        # Add synthetic columns and invalid column warnings back to the results
        if requested_synthetic_columns or invalid_columns:
            second_pass_results = self._add_synthetic_columns(
                second_pass_results,
                requested_synthetic_columns or [],
                invalid_columns,
            )

        # Ensure the results are in the same order as the IDs
        id_to_index = {id: i for i, id in enumerate(top_ids)}
        second_pass_results.sort(
            key=lambda t: id_to_index.get(t.get("id"), float("inf"))
        )

        return second_pass_results
