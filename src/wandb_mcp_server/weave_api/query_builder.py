"""
Query builder for the Weave API.

This module provides utilities for constructing query expressions for the Weave API.
It separates the complex query construction logic from the API client.
"""

import calendar
from datetime import datetime
from typing import Any, Dict, Optional, Tuple, Union

# Import the query models for building complex queries
from weave.trace_server.interface.query import (
    AndOperation,
    ContainsOperation,
    ContainsSpec,
    ConvertOperation,
    EqOperation,
    GetFieldOperator,
    GteOperation,
    GtOperation,
    LiteralOperation,
    NotOperation,
    Query,
)

from wandb_mcp_server.weave_api.models import (
    FilterOperator,
    QueryFilter,
    QueryParams,
)
from wandb_mcp_server.utils import get_rich_logger

logger = get_rich_logger(__name__)


class QueryBuilder:
    """Builds query expressions for the Weave API."""

    # Define cost fields as a class constant
    COST_FIELDS = {"total_cost", "completion_cost", "prompt_cost"}

    # Define synthetic fields that need special handling
    SYNTHETIC_FIELDS = {"status", "latency_ms"}

    @staticmethod
    def datetime_to_timestamp(dt_str: str) -> int:
        """Convert an ISO format datetime string to Unix timestamp.

        Args:
            dt_str: The ISO format datetime string. Handles 'Z' suffix for UTC.

        Returns:
            The Unix timestamp (seconds since epoch). Returns 0 if the input is empty or parsing fails.
        """
        if not dt_str:
            return 0

        # Handle 'Z' suffix for UTC
        dt_str = dt_str.replace("Z", "+00:00")

        try:
            dt = datetime.fromisoformat(dt_str)
            return int(calendar.timegm(dt.utctimetuple()))
        except ValueError:
            # If parsing fails, return 0 (beginning of epoch)
            logger.warning(f"Failed to parse datetime string: {dt_str}")
            return 0

    @classmethod
    def create_comparison_operation(
        cls, field_name: str, operator: FilterOperator, value: Any
    ) -> Optional[Union[EqOperation, GtOperation, GteOperation, NotOperation]]:
        """Create a comparison operation for a field.

        Args:
            field_name: The name of the field to compare.
            operator: The comparison operator.
            value: The value to compare against.

        Returns:
            A comparison operation, or None if the operation could not be created.
        """
        try:
            field_op_base = GetFieldOperator(**{"$getField": field_name})
            field_op = field_op_base  # Default to no $convert

            # Apply $convert selectively as it was found to be needed for some attributes
            if field_name.startswith("attributes."):
                # For general attributes, convert to double if comparing with a number
                if isinstance(value, (int, float)):
                    field_op = ConvertOperation(
                        **{"$convert": {"input": field_op_base, "to": "double"}}
                    )

            literal_op = LiteralOperation(**{"$literal": value})
        except Exception as e:
            logger.warning(
                f"Invalid value for {field_name} comparison {operator}: {value}. Error: {e}"
            )
            return None

        if operator == FilterOperator.GREATER_THAN:
            return GtOperation(**{"$gt": (field_op, literal_op)})
        elif operator == FilterOperator.GREATER_THAN_EQUAL:
            return GteOperation(**{"$gte": (field_op, literal_op)})
        elif operator == FilterOperator.EQUALS:
            return EqOperation(**{"$eq": (field_op, literal_op)})
        elif operator == FilterOperator.LESS_THAN:  # Implement $lt as $not($gte)
            gte_op = GteOperation(**{"$gte": (field_op, literal_op)})
            return NotOperation(**{"$not": [gte_op]})
        elif operator == FilterOperator.LESS_THAN_EQUAL:  # Implement $lte as $not($gt)
            gt_op = GtOperation(**{"$gt": (field_op, literal_op)})
            return NotOperation(**{"$not": [gt_op]})
        else:
            logger.warning(
                f"Unsupported comparison operator '{operator}' for {field_name}"
            )
            return None

    @classmethod
    def create_contains_operation(
        cls, field_name: str, substring: str, case_insensitive: bool = True
    ) -> ContainsOperation:
        """Create a contains operation for a field.

        Args:
            field_name: The name of the field to check.
            substring: The substring to look for.
            case_insensitive: Whether the comparison should be case-insensitive.

        Returns:
            A contains operation.
        """
        return ContainsOperation(
            **{
                "$contains": ContainsSpec(
                    input=GetFieldOperator(**{"$getField": field_name}),
                    substr=LiteralOperation(**{"$literal": substring}),
                    case_insensitive=case_insensitive,
                )
            }
        )

    @classmethod
    def build_query_expression(cls, filters: Dict[str, Any]) -> Optional[Query]:
        """Build a Query expression from the filter dictionary.

        Args:
            filters: Dictionary of filter conditions.

        Returns:
            The constructed Query object, or None if no valid filters were provided.
        """
        operations = []

        # Handle op_name filter (regex or string)
        if "op_name" in filters:
            op_name = filters["op_name"]
            if isinstance(op_name, str):
                # If it's a string with wildcard pattern, treat as contains
                if "*" in op_name or ".*" in op_name:
                    # Extract the part between wildcards
                    pattern = op_name.replace("*", "").replace(".*", "")
                    operations.append(cls.create_contains_operation("op_name", pattern))
                else:
                    # Exact match
                    operations.append(
                        EqOperation(
                            **{
                                "$eq": (
                                    GetFieldOperator(**{"$getField": "op_name"}),
                                    LiteralOperation(**{"$literal": op_name}),
                                )
                            }
                        )
                    )
            elif hasattr(op_name, "pattern"):  # Regex pattern
                operations.append(
                    cls.create_contains_operation("op_name", op_name.pattern)
                )

        # Handle op_name_contains custom filter (for simple substring matching)
        if "op_name_contains" in filters:
            substring = filters["op_name_contains"]
            operations.append(cls.create_contains_operation("op_name", substring))

        # Handle display_name filter (regex or string)
        if "display_name" in filters:
            display_name = filters["display_name"]
            if isinstance(display_name, str):
                # If it's a string with wildcard pattern, treat as contains
                if "*" in display_name or ".*" in display_name:
                    # Extract the part between wildcards
                    pattern = display_name.replace("*", "").replace(".*", "")
                    operations.append(
                        cls.create_contains_operation("display_name", pattern)
                    )
                else:
                    # Exact match
                    operations.append(
                        EqOperation(
                            **{
                                "$eq": (
                                    GetFieldOperator(**{"$getField": "display_name"}),
                                    LiteralOperation(**{"$literal": display_name}),
                                )
                            }
                        )
                    )
            elif hasattr(display_name, "pattern"):  # Regex pattern
                operations.append(
                    cls.create_contains_operation("display_name", display_name.pattern)
                )

        # Handle display_name_contains custom filter (for simple substring matching)
        if "display_name_contains" in filters:
            substring = filters["display_name_contains"]
            operations.append(cls.create_contains_operation("display_name", substring))

        # Handle status filter based on summary.weave.status using dot notation
        if "status" in filters:
            target_status = filters["status"]
            if isinstance(target_status, str):
                comp_op = cls.create_comparison_operation(
                    "summary.weave.status", FilterOperator.EQUALS, target_status.lower()
                )
                if comp_op:
                    operations.append(comp_op)
            else:
                logger.warning(
                    f"Invalid status filter value: {target_status}. Expected a string."
                )

        # Handle time range filter (convert ISO datetime strings to Unix seconds)
        if "time_range" in filters:
            time_range = filters["time_range"]

            # >= start
            if "start" in time_range and time_range["start"]:
                start_ts = cls.datetime_to_timestamp(time_range["start"])
                if start_ts > 0:
                    comp_op = cls.create_comparison_operation(
                        "started_at", FilterOperator.GREATER_THAN_EQUAL, start_ts
                    )
                    if comp_op:
                        operations.append(comp_op)

            # < end (i.e. started_at strictly before end_ts)
            if "end" in time_range and time_range["end"]:
                end_ts = cls.datetime_to_timestamp(time_range["end"])
                if end_ts > 0:
                    comp_op = cls.create_comparison_operation(
                        "started_at", FilterOperator.LESS_THAN, end_ts
                    )
                    if comp_op:
                        operations.append(comp_op)

        # Handle wb_run_id filter (top-level)
        if "wb_run_id" in filters:
            run_id = filters["wb_run_id"]
            # This filter expects a string for wb_run_id and uses $contains or $eq.
            if isinstance(run_id, str):
                if (
                    "$contains" in run_id or "*" in run_id
                ):  # Simple check for contains style
                    pattern = run_id.replace("$contains:", "").replace(
                        "*", ""
                    )  # Basic cleanup
                    operations.append(
                        cls.create_contains_operation("wb_run_id", pattern.strip())
                    )
                else:
                    operations.append(
                        EqOperation(
                            **{
                                "$eq": (
                                    GetFieldOperator(**{"$getField": "wb_run_id"}),
                                    LiteralOperation(**{"$literal": run_id}),
                                )
                            }
                        )
                    )
            elif (
                isinstance(run_id, dict) and "$contains" in run_id
            ):  # wb_run_id: {"$contains": "foo"}
                pattern = run_id["$contains"]
                if isinstance(pattern, str):
                    operations.append(
                        cls.create_contains_operation("wb_run_id", pattern)
                    )
                else:
                    logger.warning(
                        f"Invalid $contains value for wb_run_id: {pattern}. Expected string."
                    )
            else:
                logger.warning(
                    f"Invalid wb_run_id filter value: {run_id}. Expected a string or dict with $contains."
                )

        # Handle latency filter based on summary.weave.latency_ms
        if "latency" in filters:
            latency_filter = filters["latency"]
            # Extract the operator and value
            if isinstance(latency_filter, dict) and len(latency_filter) == 1:
                op_key, value = next(iter(latency_filter.items()))
                try:
                    op = FilterOperator(op_key)
                    comp_op = cls.create_comparison_operation(
                        "summary.weave.latency_ms", op, value
                    )
                    if comp_op:
                        operations.append(comp_op)
                except (ValueError, KeyError):
                    logger.warning(f"Invalid operator for latency filter: {op_key}")
            else:
                logger.warning(
                    f"Invalid format for latency filter: {latency_filter}. Expected a dict with one operator key."
                )

        # Handle attributes filter using dot notation AND supporting comparison operators
        if "attributes" in filters:
            attributes_filters = filters["attributes"]
            if isinstance(attributes_filters, dict):
                for attr_path, attr_value_or_op in attributes_filters.items():
                    full_attr_path = f"attributes.{attr_path}"

                    # Check if the value is a comparison operator dict or a literal
                    if (
                        isinstance(attr_value_or_op, dict)
                        and len(attr_value_or_op) == 1
                        and next(iter(attr_value_or_op.keys()))
                        in [op.value for op in FilterOperator]
                    ):
                        # It's a comparison operation
                        op_key, value = next(iter(attr_value_or_op.items()))
                        try:
                            op = FilterOperator(op_key)
                            comp_op = cls.create_comparison_operation(
                                full_attr_path, op, value
                            )
                            if comp_op:
                                operations.append(comp_op)
                        except (ValueError, KeyError):
                            logger.warning(
                                f"Invalid operator for attribute filter: {op_key}"
                            )
                    elif (
                        isinstance(attr_value_or_op, dict)
                        and "$contains" in attr_value_or_op
                    ):
                        # It's a contains operation
                        if isinstance(attr_value_or_op["$contains"], str):
                            operations.append(
                                cls.create_contains_operation(
                                    full_attr_path, attr_value_or_op["$contains"]
                                )
                            )
                        else:
                            logger.warning(
                                f"Invalid value for $contains on {full_attr_path}: {attr_value_or_op['$contains']}. Expected string."
                            )
                    else:
                        # Assume literal equality
                        comp_op = cls.create_comparison_operation(
                            full_attr_path, FilterOperator.EQUALS, attr_value_or_op
                        )
                        if comp_op:
                            operations.append(comp_op)
            else:
                logger.warning(
                    f"Invalid format for 'attributes' filter: {attributes_filters}. Expected a dictionary."
                )

        # Handle has_exception filter (checking top-level exception field)
        if "has_exception" in filters:
            has_exception = filters["has_exception"]
            # Skip filtering if has_exception is None (show everything)
            if has_exception is not None:
                # Create base operation that checks if exception is None (no exception case)
                base_op = EqOperation(
                    **{
                        "$eq": (
                            GetFieldOperator(**{"$getField": "exception"}),
                            LiteralOperation(**{"$literal": None}),
                        )
                    }
                )

                if has_exception:
                    # For has_exception=True: Negate the operation to get NOT NULL
                    operations.append(NotOperation(**{"$not": [base_op]}))
                else:
                    # For has_exception=False: Use the operation as is
                    operations.append(base_op)

        # Combine all operations with AND
        if operations:
            if len(operations) == 1:
                # Wrap the single operation in the Query model structure
                return Query(**{"$expr": operations[0]})
            else:
                # Wrap the AndOperation in the Query model structure
                and_op = AndOperation(**{"$and": operations})
                return Query(**{"$expr": and_op})

        return None  # No complex filters, so no Query object needed

    @classmethod
    def separate_filters(
        cls, filters: Union[Dict[str, Any], QueryFilter]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Separate filters into direct and complex filters.

        Args:
            filters: Dictionary or QueryFilter of filter conditions.

        Returns:
            Tuple of (direct_filters, complex_filters).
        """
        if not filters:
            return {}, {}

        # Convert QueryFilter to dict if needed
        if isinstance(filters, QueryFilter):
            filters_dict = {k: v for k, v in filters.__dict__.items() if v is not None}
        else:
            filters_dict = filters

        # Simple filters for CallsFilter object
        direct_filters = {}
        complex_filters = {}

        # Simple filter keys that go directly to the CallsFilter object
        simple_filter_keys = [
            "trace_roots_only",
            "op_names",
            "op_names_prefix",
            "trace_ids",
            "trace_parent_ids",
            "parent_ids",
            "call_ids",
        ]

        for key in simple_filter_keys:
            if key in filters_dict:
                # Ensure op_names, trace_ids, etc. are lists as expected by CallsFilter
                if key in [
                    "op_names",
                    "op_names_prefix",
                    "trace_ids",
                    "trace_parent_ids",
                    "parent_ids",
                    "call_ids",
                ] and not isinstance(filters_dict[key], list):
                    direct_filters[key] = [str(filters_dict[key])]
                else:
                    direct_filters[key] = filters_dict[key]

                # Ensure call_ids are strings
                if key == "call_ids" and isinstance(direct_filters[key], list):
                    direct_filters[key] = [
                        str(call_id) for call_id in direct_filters[key]
                    ]

        # Handle individual op_name and trace_id if op_names/trace_ids not already set
        if "op_name" in filters_dict and "op_names" not in direct_filters:
            # Only add if it's a simple name, not a pattern (patterns go to complex)
            if (
                isinstance(filters_dict["op_name"], str)
                and "*" not in filters_dict["op_name"]
                and ".*" not in filters_dict["op_name"]
            ):
                direct_filters["op_names"] = [filters_dict["op_name"]]
            else:
                # It's a pattern or complex, send to complex_filters
                complex_filters["op_name"] = filters_dict["op_name"]
        elif "op_name" in filters_dict and "op_names" in direct_filters:
            # If op_names is already set, and op_name is a pattern, it needs to go to complex
            if isinstance(filters_dict["op_name"], str) and (
                "*" in filters_dict["op_name"] or ".*" in filters_dict["op_name"]
            ):
                complex_filters["op_name"] = filters_dict["op_name"]

        if "trace_id" in filters_dict and "trace_ids" not in direct_filters:
            direct_filters["trace_ids"] = [str(filters_dict["trace_id"])]

        # All other keys from the original `filters` dict go to complex_filters
        all_handled_direct_keys = set(direct_filters.keys())
        # Add op_name/trace_id to handled if they were processed into op_names/trace_ids
        if (
            "op_names" in direct_filters
            and "op_name" in filters_dict
            and filters_dict["op_name"] in direct_filters["op_names"]
        ):
            all_handled_direct_keys.add("op_name")
        if (
            "trace_ids" in direct_filters
            and "trace_id" in filters_dict
            and filters_dict["trace_id"] in direct_filters["trace_ids"]
        ):
            all_handled_direct_keys.add("trace_id")

        for key, value in filters_dict.items():
            if key not in all_handled_direct_keys:
                # Exception: if op_name was simple and handled, but is also in filters, it shouldn't be added again
                if (
                    key == "op_name"
                    and "op_names" in direct_filters
                    and value in direct_filters["op_names"]
                    and not (isinstance(value, str) and ("*" in value or ".*" in value))
                ):
                    continue
                complex_filters[key] = value

        return direct_filters, complex_filters

    @classmethod
    def prepare_query_params(
        cls, params: Union[QueryParams, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Prepare query parameters for the Weave API.

        Args:
            params: Query parameters, either as a QueryParams object or a dictionary.

        Returns:
            Dictionary of query parameters ready for the Weave API.
        """
        # Convert QueryParams to dict if needed
        if isinstance(params, QueryParams):
            # Convert to dict (simplified for illustration)
            raw_params = {
                "entity_name": params.entity_name,
                "project_name": params.project_name,
                "filters": params.filters,
                "sort_by": params.sort_by,
                "sort_direction": params.sort_direction,
                "limit": params.limit,
                "offset": params.offset,
                "include_costs": params.include_costs,
                "include_feedback": params.include_feedback,
                "columns": params.columns,
                "expand_columns": params.expand_columns,
            }
        else:
            raw_params = params.copy()

        # Extract filters
        filters = raw_params.get("filters", {})

        # Separate filters
        direct_filters, complex_filters = cls.separate_filters(filters)

        # Prepare request body
        request_body = {
            "project_id": f"{raw_params['entity_name']}/{raw_params['project_name']}",
            "include_costs": raw_params.get("include_costs", True),
            "include_feedback": raw_params.get("include_feedback", True),
        }

        # Add filter if present
        if direct_filters:
            request_body["filter"] = direct_filters

        # Add query expression if present
        query_expression = cls.build_query_expression(complex_filters)
        if query_expression:
            request_body["query"] = query_expression.model_dump(by_alias=True)

        # Add sort criteria if present, but never send cost fields to server
        sort_by = raw_params.get("sort_by")
        if sort_by and sort_by not in cls.COST_FIELDS:
            request_body["sort_by"] = [
                {
                    "field": sort_by,
                    "direction": raw_params.get("sort_direction", "desc"),
                }
            ]

        # Add pagination parameters if present
        if "limit" in raw_params and raw_params["limit"] is not None:
            request_body["limit"] = raw_params["limit"]
        if "offset" in raw_params and raw_params["offset"] is not None:
            request_body["offset"] = raw_params["offset"]

        # Process columns, filtering out synthetic fields
        if "columns" in raw_params and raw_params["columns"]:
            original_columns = raw_params["columns"]

            # Store synthetic fields we need to generate later
            synthetic_fields_to_add = []

            # Filter out synthetic fields from the API request
            filtered_columns = []
            for col in original_columns:
                if col in cls.SYNTHETIC_FIELDS:
                    synthetic_fields_to_add.append(col)
                else:
                    filtered_columns.append(col)

            # If we need to synthesize status, ensure we request summary field
            if (
                "status" in synthetic_fields_to_add
                and "summary" not in filtered_columns
            ):
                filtered_columns.append("summary")

            # If we need to synthesize latency_ms, ensure we request summary field
            if (
                "latency_ms" in synthetic_fields_to_add
                and "summary" not in filtered_columns
            ):
                filtered_columns.append("summary")

            # Only send filtered columns to the API
            if filtered_columns:
                request_body["columns"] = filtered_columns

            # Store the synthetic fields that need to be processed
            if synthetic_fields_to_add:
                request_body["_synthetic_fields"] = synthetic_fields_to_add

        if "expand_columns" in raw_params and raw_params["expand_columns"]:
            request_body["expand_columns"] = raw_params["expand_columns"]

        return request_body
