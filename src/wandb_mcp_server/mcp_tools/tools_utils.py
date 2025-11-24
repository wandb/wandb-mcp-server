import inspect
import re
from typing import Any, Callable, Dict, Type, Union, Tuple, Optional
from wandb_mcp_server.utils import get_rich_logger
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# _map_python_type_to_json_schema remains the same...
def _map_python_type_to_json_schema(py_type: Type[Any]) -> Dict[str, Any]:
    """Maps Python types to JSON Schema type definitions."""
    origin = getattr(py_type, "__origin__", None)
    args = getattr(py_type, "__args__", ())

    if py_type is str:
        return {"type": "string"}
    elif py_type is int:
        return {"type": "integer"}
    elif py_type is float:
        return {"type": "number"}
    elif py_type is bool:
        return {"type": "boolean"}
    elif py_type is list or origin is list:
        items_schema = {"type": "string"}  # Default for untyped lists
        if args and args[0] is not Any:
            items_schema = _map_python_type_to_json_schema(args[0])
        return {"type": "array", "items": items_schema}
    elif py_type is dict or origin is dict:
        # Basic object type, doesn't specify properties for simplicity
        # Might need refinement if nested structures are common and need schema
        return {"type": "object", "additionalProperties": True}  # Allow any properties
    elif origin is Union:
        # Handle Optional[T] -> T (represented as Union[T, NoneType])
        # And basic Union types, favouring the first non-None type
        non_none_types = [t for t in args if t is not type(None)]
        if len(non_none_types) > 0:
            # If it was Optional[T], is_optional flag handles 'required' status later
            # For Union[A, B], just use the first type's schema for simplicity.
            # A more complex approach could use 'anyOf'.
            return _map_python_type_to_json_schema(non_none_types[0])
        else:
            return {"type": "null"}  # Should only happen for Union[NoneType]
    elif py_type is Any:
        # Any type can be represented loosely, e.g., allowing any type
        return {}  # Represents any type in JSON Schema when empty
    else:
        # Default or unknown types
        # Consider logging a warning for unknown types
        return {"type": "string"}  # Default to string if unknown


def _parse_docstring(docstring: str) -> Tuple[Dict[str, str], str]:
    """
    Parses a NumPy-style docstring to extract the main description and parameter descriptions.
    Handles "Parameters", "Args:", "Arguments:" sections.
    """
    if not docstring:
        return {}, ""

    lines = docstring.strip().splitlines()
    main_description_lines = []
    param_descriptions = {}

    # 1. Find main description and Parameter section variants
    potential_headers = {"parameters", "args:", "arguments:"}
    header_found = False
    param_definitions_start_index = -1

    for i, line in enumerate(lines):
        stripped_line = line.strip()
        stripped_lower = stripped_line.lower()

        if not header_found:
            if stripped_lower in potential_headers:
                header_found = True
                # Check for optional '---' separator line immediately after
                if i + 1 < len(lines) and lines[i + 1].strip().startswith("---"):
                    param_definitions_start_index = i + 2
                else:
                    param_definitions_start_index = i + 1
                # Stop adding to main description *before* the header line
            elif stripped_line:
                main_description_lines.append(stripped_line)
        elif i >= param_definitions_start_index:
            # Break early if we found the header and are past the definition start line
            # The next loop will handle parsing from here
            break

    main_description = "\n".join(main_description_lines).strip()

    # 2. Process Parameters section if found
    if param_definitions_start_index != -1 and param_definitions_start_index < len(
        lines
    ):
        current_param_name = None
        current_param_desc_lines = []
        param_def_indent = -1
        expected_desc_indent = -1

        for i in range(param_definitions_start_index, len(lines)):
            line = lines[i]
            current_indent = len(line) - len(line.lstrip(" "))
            stripped_line = line.strip()

            # Regex to find 'name :' at the start of the stripped line
            param_match = re.match(
                r"^(?P<name>[a-zA-Z_]\w*)\s*:(?P<desc_start>.*)$", stripped_line
            )

            # --- Determine if this line starts a new parameter ---
            is_new_parameter_line = False
            if param_match:
                # It looks like a parameter definition. Is it *really* a new one?
                # It's new if:
                #   a) We aren't currently processing a param OR
                #   b) It's not indented further than the *start* of the current param's description block
                #      (to allow param names within descriptions, although uncommon in NumPy style) OR
                #   c) It's strictly less indented than the expected description indent (clearly ends the block) OR
                #   d) It's at the same or lesser indent level than the previous param def line itself.
                if not current_param_name:
                    is_new_parameter_line = True
                else:
                    if expected_desc_indent != -1:
                        # If we have an established description indent
                        if current_indent < expected_desc_indent:
                            is_new_parameter_line = (
                                True  # Definitely ends previous block
                            )
                    elif current_indent <= param_def_indent:
                        # If no description started yet, or if back at original indent
                        is_new_parameter_line = True

            # --- Process based on whether it's a new parameter line ---
            if is_new_parameter_line:
                # Save the previous parameter's description
                if current_param_name:
                    param_descriptions[current_param_name] = "\n".join(
                        current_param_desc_lines
                    ).strip()

                # Start the new parameter
                current_param_name = param_match.group("name")
                desc_start = param_match.group("desc_start").strip()
                current_param_desc_lines = (
                    [desc_start] if desc_start else []
                )  # Use first line if present
                param_def_indent = current_indent
                expected_desc_indent = -1  # Reset for the new parameter

            elif current_param_name:
                # We are inside a parameter's block, and this line is not starting a new param
                if not stripped_line:
                    # Handle blank lines within the description
                    # Keep blank lines if they are indented same/more than expected desc indent OR
                    # if they are more indented than param def and we haven't set expected yet.
                    if (
                        expected_desc_indent != -1
                        and current_indent >= expected_desc_indent
                    ) or (
                        expected_desc_indent == -1 and current_indent > param_def_indent
                    ):
                        current_param_desc_lines.append("")  # Preserve paragraph breaks
                    # Otherwise, ignore blank lines that break indentation pattern
                    continue

                # This is a non-empty, non-param-starting line within a block
                if expected_desc_indent == -1:
                    # This is the first line of the description text (after the 'name :' line)
                    if current_indent > param_def_indent:
                        expected_desc_indent = current_indent
                        current_param_desc_lines.append(stripped_line)
                    else:
                        # Indentation is not correct for a description start. End of this param.
                        if current_param_name:  # Save description if any collected
                            param_descriptions[current_param_name] = "\n".join(
                                current_param_desc_lines
                            ).strip()
                        current_param_name = None
                        # Assume end of parameters section, stop parsing this section
                        break
                elif current_indent >= expected_desc_indent:
                    # Continuation of the description (matches or exceeds expected indent)
                    current_param_desc_lines.append(stripped_line)
                else:
                    # Indentation decreased below expected description indent. End of this parameter's description.
                    if current_param_name:
                        param_descriptions[current_param_name] = "\n".join(
                            current_param_desc_lines
                        ).strip()
                    current_param_name = None
                    # Assume end of parameters section, stop parsing this section
                    break
            else:
                # We are not inside a parameter block, and this line doesn't start one.
                # If the line has content, it must be the end of the Parameters section.
                if stripped_line:
                    break  # Stop parsing parameters section

        # Save the last parameter description after the loop finishes
        if current_param_name:
            param_descriptions[current_param_name] = "\n".join(
                current_param_desc_lines
            ).strip()

    # Filter out empty descriptions that might result from parsing issues or empty entries
    param_descriptions = {k: v for k, v in param_descriptions.items() if v}

    return param_descriptions, main_description


# generate_anthropic_tool_schema remains the same, but needs slight adjustment for required logic
def generate_anthropic_tool_schema(
    func: Callable[..., Any], description: str | None = None
) -> Dict[str, Any]:
    """
    Generates an Anthropic tool schema dictionary from a Python function.

    Args:
        func: The function to generate the schema for.
        description: Optional override for the tool description. If None, it's parsed from the docstring.
    """
    signature = inspect.signature(func)
    docstring = inspect.getdoc(func) or ""

    param_docs, main_description = _parse_docstring(docstring)

    # Use provided description if available, otherwise use parsed main description
    final_description = description if description is not None else main_description

    properties = {}
    required_params = []

    for name, param in signature.parameters.items():
        if name in ("self", "cls"):  # Skip self/cls for methods
            continue

        schema = {}
        param_type = param.annotation

        # --- Type Mapping ---
        if param_type is not inspect.Parameter.empty:
            type_schema = _map_python_type_to_json_schema(param_type)
            schema.update(type_schema)
        else:
            # Default to string if no type hint
            schema["type"] = "string"

        # --- Description ---
        # Get description from parsed docstring, fallback to empty string
        schema["description"] = param_docs.get(name, "").strip()

        # --- Required Status & Default ---
        is_optional = False
        if param.default is not inspect.Parameter.empty:
            # Has a default value, so not required by default
            if param.default is not None:  # Append default only if not None
                default_str = f"Default: {param.default!r}."
                if schema["description"]:
                    schema["description"] += f" {default_str}"
                else:
                    schema["description"] = default_str
        else:
            # No default value. Check if type hint marks it as Optional
            origin = getattr(param_type, "__origin__", None)
            args = getattr(param_type, "__args__", ())
            if origin is Union and type(None) in args:
                is_optional = True  # Type hint is Optional[T] or Union[T, None]
            elif param_type is Any:
                # Assume Any could be None, treat as not strictly required unless logic dictates otherwise
                is_optional = True  # Or base this on project conventions for 'Any'
            elif param_type is inspect.Parameter.empty:
                # No type hint, no default -> Assume required
                pass  # is_optional remains False
            # If not Optional or Any without default, it's required
            if not is_optional:
                required_params.append(name)

        properties[name] = schema

    tool_schema = {
        "name": func.__name__,
        "description": final_description,  # Use the determined description
        "input_schema": {
            "type": "object",
            "properties": properties,
        },
    }
    # Only add 'required' key if there are required parameters
    if required_params:
        # Ensure uniqueness and sort for consistency
        tool_schema["input_schema"]["required"] = sorted(list(set(required_params)))

    return tool_schema


def get_retry_session(
    retries: int = 3,
    backoff_factor: float = 1.0,
    status_forcelist: Tuple[int, ...] = (429, 500, 502, 503, 504),
    allowed_methods: Tuple[str, ...] = ("POST", "GET"),
) -> requests.Session:
    """Get a requests session with retry capabilities.

    Args:
        retries: Total number of retries to allow.
        backoff_factor: A backoff factor to apply between attempts after the second try.
                        {backoff factor} * (2 ** ({number of total retries} - 1))
        status_forcelist: A set of HTTP status codes that we should force a retry on.
        allowed_methods: A list of uppercase HTTP method verbs that we should allow retries on.

    Returns:
        A requests.Session object configured with retry logic.
    """
    session = requests.Session()
    retry_strategy = Retry(
        total=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        allowed_methods=list(allowed_methods),  # Retry expects a list
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def log_tool_call(tool_name: str, viewer: Any, params: Dict[str, Any]) -> None:
    """
    Minimal helper to log tool calls consistently across mcp_tools.

    No truncation/redaction.
    """
    logger = get_rich_logger("mcp_tools")
    try:
        logger.info(f"ToolCall name={tool_name} viewer={viewer} params={params}")
    except Exception:
        # Never fail tool execution due to logging
        pass