"""Module for querying the W&B GraphQL API."""

import copy
import logging
import re
import traceback
from typing import Any, Dict, List, Optional

from graphql import parse
from graphql.language import ast as gql_ast
from graphql.language import printer as gql_printer
from graphql.language import visitor as gql_visitor
from wandb_mcp_server.mcp_tools.tools_utils import track_tool_execution
from wandb_mcp_server.utils import get_rich_logger
from wandb_mcp_server.wandb_graphql import (
    GraphQLReadOnlyViolation,
    execute_graphql,
    validate_read_only_graphql,
)

logger = get_rich_logger(__name__)


QUERY_WANDB_GRAPHQL_TOOL_DESCRIPTION = """Execute an advanced, query-only GraphQL document against W&B Models.

This opt-in escape hatch is only for reads without public W&B SDK parity:
- schema introspection
- unmodeled or custom fields
- cross-resource nested selections
- aliases or an exact GraphQL response shape

<when_to_use>
Use only when the requested read cannot be represented by the public W&B SDK and
the deployment administrator has deliberately enabled raw GraphQL access.
</when_to_use>

Do not use this tool for projects, run lookup/filtering/sorting, sweeps, reports,
artifacts, registries, automations, integrations, or run history. Those operations
have SDK-backed MCP tools and should use them instead.

Only GraphQL query operations are accepted. Mutations, subscriptions, and mixed
documents are rejected before a W&B request. Collection pagination requires the
W&B edges/node/pageInfo connection shape.

Parameters
----------
query : str
    A complete read-only GraphQL query document.
variables : dict, optional
    Variables referenced by the document.
max_items : int, optional
    Maximum items accumulated across pages. Default: 100.
items_per_page : int, optional
    Requested collection page size. Default: 20.
"""


def find_paginated_collections(obj: Dict, current_path: Optional[List[str]] = None) -> List[List[str]]:
    """Find collections in a response that follow the W&B connection pattern. Returns List[List[str]]."""
    # Ensure this implementation correctly builds and returns List[List[str]]
    if current_path is None:
        current_path = []
    collections = []
    if isinstance(obj, dict):
        if (
            "edges" in obj
            and "pageInfo" in obj
            and isinstance(obj.get("edges"), list)
            and isinstance(obj.get("pageInfo"), dict)
            and "hasNextPage" in obj.get("pageInfo", {})
            and "endCursor" in obj.get("pageInfo", {})
        ):
            collections.append(list(current_path))  # Correct: append list path
        # Recurse correctly
        for key, value in obj.items():
            current_path.append(key)
            collections.extend(find_paginated_collections(value, current_path))
            current_path.pop()
    elif isinstance(obj, list):
        for item in obj:
            collections.extend(find_paginated_collections(item, current_path))
    return collections


def get_nested_value(obj: Dict, path: list[str]) -> Optional[Any]:
    """Get a value from a nested dictionary using a list of keys (path)."""
    current = obj
    # Iterate directly over the list path
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


_HOSTED_LIMIT_VARIABLE_RE = re.compile(r"^(first|limit|count|max_?items|page_?size|items_per_page)$", re.I)


def _field_name(node: gql_ast.FieldNode) -> str:
    """Return the response field name for a GraphQL field."""
    return node.alias.value if node.alias else node.name.value


def _is_connection_field(node: gql_ast.FieldNode) -> bool:
    """Return True if a field selection looks like a W&B connection."""
    if not node.selection_set:
        return False
    child_names = {
        selection.name.value for selection in node.selection_set.selections if isinstance(selection, gql_ast.FieldNode)
    }
    return {"edges", "pageInfo"}.issubset(child_names)


class HostedGraphQLPreflightVisitor(gql_visitor.Visitor):
    """Clamp hosted GraphQL page sizes and reject unsafe fanout shapes."""

    def __init__(self, max_first: int) -> None:
        super().__init__()
        self.max_first = max_first
        self.path: list[str] = []
        self.connection_depth = 0
        self.connection_paths: list[tuple[str, ...]] = []
        self.rewrites: list[str] = []
        self.rejections: list[str] = []

    def enter_variable_definition(self, node, key, parent, path, ancestors):
        if not isinstance(node, gql_ast.VariableDefinitionNode):
            return
        if not _HOSTED_LIMIT_VARIABLE_RE.match(node.variable.name.value):
            return
        if not isinstance(node.default_value, gql_ast.IntValueNode):
            return

        requested = int(node.default_value.value)
        if requested <= self.max_first:
            return

        node.default_value = gql_ast.IntValueNode(value=str(self.max_first))
        self.rewrites.append(f"Clamped ${node.variable.name.value} default from {requested} to {self.max_first}")

    def enter_field(self, node, key, parent, path, ancestors):
        if not isinstance(node, gql_ast.FieldNode):
            return

        self.path.append(_field_name(node))
        if not _is_connection_field(node):
            return

        current_path = tuple(self.path)
        self.connection_paths.append(current_path)
        if self.connection_depth > 0:
            self.rejections.append(
                f"Nested paginated collection is not allowed in hosted mode: {'/'.join(current_path)}"
            )
        if len(self.connection_paths) > 1:
            self.rejections.append(
                f"Hosted GraphQL queries may include only one paginated collection; found {len(self.connection_paths)}."
            )

        existing_args = list(node.arguments or [])
        has_first = False
        for idx, arg in enumerate(existing_args):
            arg_name = arg.name.value
            if arg_name == "last":
                self.rejections.append(
                    f"Hosted GraphQL does not support reverse pagination with last: {'/'.join(current_path)}"
                )
            if arg_name != "first":
                continue
            has_first = True
            if isinstance(arg.value, gql_ast.IntValueNode):
                requested = int(arg.value.value)
                if requested > self.max_first:
                    existing_args[idx] = gql_ast.ArgumentNode(
                        name=arg.name,
                        value=gql_ast.IntValueNode(value=str(self.max_first)),
                    )
                    self.rewrites.append(f"Clamped {'/'.join(current_path)} first from {requested} to {self.max_first}")

        if not has_first:
            self.rejections.append(
                "Hosted GraphQL paginated collections must include a first argument so the initial request "
                f"can be bounded: {'/'.join(current_path)}"
            )

        node.arguments = tuple(existing_args)
        self.connection_depth += 1

    def leave_field(self, node, key, parent, path, ancestors):
        if isinstance(node, gql_ast.FieldNode) and _is_connection_field(node):
            self.connection_depth = max(0, self.connection_depth - 1)
        if self.path:
            self.path.pop()


def _hosted_preprocess_graphql_query(query: str, max_first: int) -> tuple[str, list[str], list[str]]:
    """Rewrite hosted GraphQL pagination limits before the first execution."""
    document = parse(query.strip())
    visitor = HostedGraphQLPreflightVisitor(max_first=max_first)
    rewritten = gql_visitor.visit(document, visitor)
    return gql_printer.print_ast(rewritten), visitor.rewrites, visitor.rejections


def _hosted_clamp_variables(variables: Dict[str, Any], max_first: int) -> Dict[str, Any]:
    """Clamp common page-size variable names before the initial GraphQL execute."""
    clamped = dict(variables)
    for key, value in list(clamped.items()):
        if not _HOSTED_LIMIT_VARIABLE_RE.match(key):
            continue
        if isinstance(value, int) and value > max_first:
            clamped[key] = max_first
    return clamped


def query_paginated_wandb_gql(
    query: str,
    variables: Optional[Dict[str, Any]] = None,
    max_items: int = 100,
    items_per_page: int = 50,
) -> Dict[str, Any]:
    """
    Execute a GraphQL query against the W&B API with pagination support using AST modification.
    Handles a single paginated field detected via the connection pattern.
    Modifies the result dictionary in-place.

    Args:
        query: The GraphQL query string. MUST include pageInfo{hasNextPage, endCursor} for paginated fields.
        variables: Variables to pass to the GraphQL query.
        max_items: Maximum number of items to fetch across all pages (default: 100).
        items_per_page: Number of items to request per page (default: 20).
        deduplicate: Whether to deduplicate nodes by ID across pages (default: True).

    Returns:
        The aggregated GraphQL response dictionary.
    """
    try:
        validate_read_only_graphql(query)
    except GraphQLReadOnlyViolation as e:
        return {
            "errors": [
                {
                    "error": "read_only_violation",
                    "message": str(e),
                    "operation_types": list(e.operation_types),
                }
            ]
        }
    except Exception as e:
        return {"errors": [{"message": f"Failed to validate initial query: {e}"}]}

    from wandb_mcp_server.api_client import get_wandb_api

    api = get_wandb_api()
    result_dict = {}
    limit_key = None
    with track_tool_execution(
        "query_paginated_wandb_gql",
        None,
        {
            "query": query,
            "variables": variables,
            "max_items": max_items,
            "items_per_page": items_per_page,
        },
        mcp_tool_name="query_wandb_graphql_tool",
    ) as ctx:
        try:
            from wandb_mcp_server.config import (
                MCP_HOSTED_MODE,
                MCP_MAX_GQL_ITEMS,
                MCP_MAX_GQL_ITEMS_PER_PAGE,
            )

            if MCP_HOSTED_MODE:
                if max_items > MCP_MAX_GQL_ITEMS:
                    logger.warning("Clamping hosted GraphQL max_items from %s to %s", max_items, MCP_MAX_GQL_ITEMS)
                    max_items = MCP_MAX_GQL_ITEMS
                if items_per_page > MCP_MAX_GQL_ITEMS_PER_PAGE:
                    logger.warning(
                        "Clamping hosted GraphQL items_per_page from %s to %s",
                        items_per_page,
                        MCP_MAX_GQL_ITEMS_PER_PAGE,
                    )
                    items_per_page = MCP_MAX_GQL_ITEMS_PER_PAGE

            logger.info("--- Inside query_paginated_wandb_gql: Step 0: Execute Initial Query ---")

            page1_vars_func = variables.copy() if variables is not None else {}
            limit_key = None
            for k in page1_vars_func:
                if k.lower() in ["limit", "first", "count"]:
                    limit_key = k
                    break
            if limit_key:
                page1_vars_func[limit_key] = min(items_per_page, page1_vars_func.get(limit_key) or items_per_page)
            else:
                limit_key = "limit"
                page1_vars_func[limit_key] = items_per_page
                logger.debug(f"No limit variable found in input, adding '{limit_key}={items_per_page}'")

            if MCP_HOSTED_MODE:
                try:
                    query, rewrites, rejections = _hosted_preprocess_graphql_query(
                        query,
                        MCP_MAX_GQL_ITEMS_PER_PAGE,
                    )
                    page1_vars_func = _hosted_clamp_variables(
                        page1_vars_func,
                        MCP_MAX_GQL_ITEMS_PER_PAGE,
                    )
                    if rejections:
                        ctx.mark_error(f"query_too_complex: {rejections[0]}")
                        return {
                            "errors": [
                                {
                                    "message": rejections[0],
                                    "details": rejections,
                                    "error": "query_too_complex",
                                }
                            ]
                        }
                    if rewrites:
                        logger.warning("Hosted GraphQL preflight rewrote query: %s", rewrites)
                except Exception as e:
                    logger.error("Hosted GraphQL preflight failed: %s", e, exc_info=True)
                    ctx.mark_error(f"invalid_input: {e}")
                    return {"errors": [{"message": f"Failed to validate initial query: {e}"}]}

            try:
                parse(query.strip())
            except Exception as e:
                logger.error(f"Failed to parse initial query with graphql-core: {e}")
                ctx.mark_error(f"invalid_input: {e}")
                return {"errors": [{"message": f"Failed to parse initial query: {e}"}]}

            try:
                result1 = execute_graphql(api, query.strip(), page1_vars_func)
                result_dict = copy.deepcopy(result1)
                if "errors" in result_dict:
                    logger.error(f"GraphQL errors in initial response: {result_dict['errors']}")
                    ctx.mark_error("upstream_error: GraphQL errors in initial response")
                    return result_dict
            except Exception as e:
                logger.error(f"Failed to execute initial GraphQL query: {e}", exc_info=True)
                ctx.mark_error(f"upstream_error: {e}")
                return {"errors": [{"message": f"Failed to execute initial query: {e}"}]}

            detected_paths = find_paginated_collections(result_dict)
            if not detected_paths:
                logger.info("No paginated paths detected. Returning initial result.")
                return result_dict

            path_to_paginate = detected_paths[0]
            logger.info(f"Using path for pagination: {'/'.join(path_to_paginate)}")

            runs_data1 = get_nested_value(result_dict, path_to_paginate)
            if runs_data1 is None:
                logger.warning(
                    f"Could not extract data for pagination path {'/'.join(path_to_paginate)}. Returning initial result."
                )
                return result_dict
            page_info1 = get_nested_value(runs_data1, ["pageInfo"])
            if page_info1 is None:
                logger.warning(
                    f"Could not extract pageInfo for pagination path {'/'.join(path_to_paginate)}. Returning initial result."
                )
                return result_dict

            cursor = page_info1.get("endCursor")
            has_next = page_info1.get("hasNextPage")
            initial_edges = runs_data1.get("edges", [])
            logging.info(f"Page 1 Results: {len(initial_edges)} runs.")
            logging.info(f"Page 1 PageInfo: {page_info1}")

            seen_ids = set()
            current_edge_count = 0
            temp_initial_edges = []
            if initial_edges:
                for edge in initial_edges:
                    try:
                        if current_edge_count >= max_items:
                            break
                        node_id = edge["node"]["id"]
                        if node_id not in seen_ids:
                            seen_ids.add(node_id)
                            temp_initial_edges.append(edge)
                            current_edge_count += 1
                    except (KeyError, TypeError):
                        if current_edge_count < max_items:
                            temp_initial_edges.append(edge)
                            current_edge_count += 1
                target_collection_dict = get_nested_value(result_dict, path_to_paginate)
                if target_collection_dict:
                    target_collection_dict["edges"] = temp_initial_edges[:max_items]
                    current_edge_count = len(target_collection_dict["edges"])
                logging.info(f"Stored {current_edge_count} unique edges after page 1 (max: {max_items}).")

            if not has_next or not cursor or current_edge_count >= max_items:
                logger.info("No further pages needed based on page 1 info or max_items reached.")
                target_pi_dict = get_nested_value(result_dict, path_to_paginate + ["pageInfo"])
                if target_pi_dict:
                    target_pi_dict["hasNextPage"] = False
                return result_dict

            logging.info("\n--- Generating Paginated Query String --- ")
            generated_paginated_query_string = None
            after_variable_name = "after"
            try:
                initial_ast = parse(query.strip())
                visitor = AddPaginationArgsVisitor(
                    field_paths=detected_paths,
                    first_variable_name=limit_key,
                    after_variable_name=after_variable_name,
                )
                modified_ast = gql_visitor.visit(copy.deepcopy(initial_ast), visitor)
                generated_paginated_query_string = gql_printer.print_ast(modified_ast)
                logger.info("AST modification and printing successful.")
            except Exception as e:
                logger.error(f"Failed to generate query string via AST: {e}", exc_info=True)
                return result_dict

            if generated_paginated_query_string is None:
                return result_dict

            logging.info("\n--- Loop: Execute, Deduplicate, Aggregate In-Place, Check Limit ---")
            page_num = 1
            current_cursor = cursor
            current_has_next = has_next
            final_page_info = page_info1

            while current_has_next:
                if current_edge_count >= max_items:
                    logging.info(f"Reached max_items ({max_items}). Stopping loop.")
                    final_page_info = {**final_page_info, "hasNextPage": False}
                    break

                page_num += 1
                logging.info(f"\nFetching Page {page_num}...")
                page_vars = variables.copy() if variables is not None else {}
                page_vars[limit_key] = items_per_page
                page_vars[after_variable_name] = current_cursor

                try:
                    logging.info(f"Executing generated query for page {page_num} with vars: {page_vars}")
                    result_page = execute_graphql(api, generated_paginated_query_string, page_vars)

                    if "errors" in result_page:
                        logger.error(
                            f"GraphQL errors on page {page_num}: {result_page['errors']}. Stopping pagination."
                        )
                        current_has_next = False
                        final_page_info = {**final_page_info, "hasNextPage": False}
                        continue

                    runs_data = get_nested_value(result_page, path_to_paginate)
                    if runs_data is None:
                        logging.warning(
                            f"Could not get data for path {'/'.join(path_to_paginate)} on page {page_num}. Stopping."
                        )
                        current_has_next = False
                        continue
                    else:
                        edges_this_page = get_nested_value(runs_data, ["edges"]) or []
                        page_info = get_nested_value(runs_data, ["pageInfo"]) or {}
                        final_page_info = page_info

                    logging.info(f"Result (Page {page_num}): {len(edges_this_page)} runs returned.")
                    logging.info(f"Page Info (Page {page_num}): {page_info}")

                    new_edges_for_aggregation = []
                    duplicates_skipped = 0
                    if edges_this_page:
                        for edge in edges_this_page:
                            if current_edge_count + len(new_edges_for_aggregation) >= max_items:
                                logging.info(f"Max items ({max_items}) reached mid-page {page_num}.")
                                final_page_info = {**final_page_info, "hasNextPage": False}
                                current_has_next = False
                                break

                            try:
                                node_id = edge["node"]["id"]
                                if node_id not in seen_ids:
                                    seen_ids.add(node_id)
                                    new_edges_for_aggregation.append(edge)
                                else:
                                    duplicates_skipped += 1
                            except (KeyError, TypeError):
                                new_edges_for_aggregation.append(edge)

                        if duplicates_skipped > 0:
                            logging.info(f"Skipped {duplicates_skipped} duplicate edges on page {page_num}.")

                        if new_edges_for_aggregation:
                            target_collection_dict_inplace = get_nested_value(result_dict, path_to_paginate)
                            if target_collection_dict_inplace and isinstance(
                                target_collection_dict_inplace.get("edges"), list
                            ):
                                target_collection_dict_inplace["edges"].extend(new_edges_for_aggregation)
                                current_edge_count = len(target_collection_dict_inplace["edges"])
                                logging.info(
                                    f"Appended {len(new_edges_for_aggregation)} new edges. Total unique edges: {current_edge_count}"
                                )
                            else:
                                logging.error("Could not find target edges list in result_dict to append in-place.")
                                current_has_next = False
                        else:
                            if len(edges_this_page) > 0:
                                logging.info("No new unique edges found on page {page_num} after deduplication.")
                            else:
                                logging.info("No edges returned on page {page_num} to aggregate.")
                    else:
                        logging.info("No edges returned on page {page_num} to aggregate.")

                    current_cursor = final_page_info.get("endCursor")
                    if current_has_next:
                        current_has_next = final_page_info.get("hasNextPage", False)

                    if current_has_next and not current_cursor:
                        logging.warning("hasNextPage is true but no endCursor received. Stopping loop.")
                        current_has_next = False
                    if not edges_this_page:
                        logging.warning(f"No edges received for page {page_num}. Stopping loop.")
                        current_has_next = False

                except Exception as e:
                    logging.error(f"Execution failed for page {page_num}: {e}", exc_info=True)
                    current_has_next = False

            logging.info(f"\n--- Pagination Loop Finished after page {page_num} ---")
            logging.info(f"Final aggregated edge count: {current_edge_count}")

            target_collection_dict_final = get_nested_value(result_dict, path_to_paginate)
            if target_collection_dict_final:
                target_collection_dict_final["pageInfo"] = final_page_info
                logging.info(f"Updated final pageInfo: {final_page_info}")

            return result_dict

        except Exception as e:
            error_message = f"Critical error in paginated GraphQL query function: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_message)
            ctx.mark_error(f"query_failed: {e}")
            if result_dict:
                if "errors" not in result_dict:
                    result_dict["errors"] = []
                result_dict["errors"].append({"message": "Pagination failed", "details": str(e)})
                return result_dict
            else:
                return {"errors": [{"message": "Pagination failed catastrophically", "details": str(e)}]}


class AddPaginationArgsVisitor(gql_visitor.Visitor):
    """Adds first/after args and variables"""

    def __init__(self, field_paths, first_variable_name="limit", after_variable_name="after"):
        super().__init__()
        self.field_paths = set(tuple(p) for p in field_paths)
        self.first_variable_name = first_variable_name
        self.after_variable_name = after_variable_name
        self.current_path = []
        self.modified_operation = False

    def enter_field(self, node, key, parent, path, ancestors):
        field_name = node.alias.value if node.alias else node.name.value
        self.current_path.append(field_name)
        current_path_tuple = tuple(self.current_path)
        if current_path_tuple in self.field_paths:
            existing_args = list(node.arguments)
            args_changed = False
            has_first = any(arg.name.value == "first" for arg in existing_args)
            if not has_first:
                # Defaulting variable name to 'limit' if not found, might need refinement
                limit_var_node = gql_ast.VariableNode(name=gql_ast.NameNode(value=self.first_variable_name))
                existing_args.append(gql_ast.ArgumentNode(name=gql_ast.NameNode(value="first"), value=limit_var_node))
                args_changed = True
            has_after = any(arg.name.value == "after" for arg in existing_args)
            if not has_after:
                existing_args.append(
                    gql_ast.ArgumentNode(
                        name=gql_ast.NameNode(value="after"),
                        value=gql_ast.VariableNode(name=gql_ast.NameNode(value=self.after_variable_name)),
                    )
                )
                args_changed = True
            if args_changed:
                node.arguments = tuple(existing_args)

    def leave_field(self, node, key, parent, path, ancestors):
        if self.current_path:
            self.current_path.pop()

    def enter_operation_definition(self, node, key, parent, path, ancestors):
        if self.modified_operation:
            return
        existing_vars = {var.variable.name.value for var in node.variable_definitions}
        new_defs_list = list(node.variable_definitions)
        defs_changed = False
        # Determine limit variable name from existing vars if possible, else default
        current_limit_var = self.first_variable_name  # Default
        for var_name in existing_vars:
            if var_name.lower() in ["limit", "first", "count"]:
                current_limit_var = var_name
                break

        if current_limit_var not in existing_vars:
            new_defs_list.append(
                gql_ast.VariableDefinitionNode(
                    variable=gql_ast.VariableNode(name=gql_ast.NameNode(value=current_limit_var)),
                    type=gql_ast.NamedTypeNode(name=gql_ast.NameNode(value="Int")),
                )
            )
            defs_changed = True
        if self.after_variable_name not in existing_vars:
            new_defs_list.append(
                gql_ast.VariableDefinitionNode(
                    variable=gql_ast.VariableNode(name=gql_ast.NameNode(value=self.after_variable_name)),
                    type=gql_ast.NamedTypeNode(name=gql_ast.NameNode(value="String")),
                )
            )
            defs_changed = True
        if defs_changed:
            node.variable_definitions = tuple(new_defs_list)
        self.modified_operation = True
