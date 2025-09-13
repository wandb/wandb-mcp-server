# import pytest
# import re
# import inspect
# import ast
# import json
# import os
# from src.wandb_mcp_server.server import query_wandb_tool # Assuming src is importable


# # --- Configuration ---
# TARGET_ENTITY = "c-metrics"
# TARGET_PROJECT = "hallucination"

# # --- Helper Function to Extract Examples ---

# def extract_gql_examples_from_docstring(docstring):
#     """Parses a docstring to extract GraphQL examples marked by specific delimiters."""
#     examples = []
#     # Regex to find the blocks delimited by <!-- WANDB_GQL_EXAMPLE_START/END -->
#     # Restore original regex with backreference
#     example_pattern = re.compile(
#         r'<!-- WANDB_GQL_EXAMPLE_START name=(\w+) -->(.*?)<!-- WANDB_GQL_EXAMPLE_END name=\1 -->', # Restored \1
#         re.DOTALL
#     )
#     # Regex to find graphql code blocks
#     graphql_pattern = re.compile(r'\s*```graphql\s*\n(.*?)\n\s*```', re.DOTALL)
#     # Regex to find python code blocks
#     python_pattern = re.compile(r'\s*```python\s*\n(.*?)\n\s*```', re.DOTALL)

#     # --- DEBUGGING ---
#     print(f"\n>>> DEBUG: Inside extract_gql_examples_from_docstring")
#     print(f"    Attempting to find matches with pattern: {example_pattern.pattern}")
#     print(f"    in docstring of length {len(docstring)}")
#     matches_found = 0
#     # --- END DEBUGGING ---

#     for match in example_pattern.finditer(docstring):
#         # --- DEBUGGING ---
#         matches_found += 1
#         print(f"    >>> Found match {matches_found}: name='{match.group(1)}'")
#         # --- END DEBUGGING ---

#         name = match.group(1)
#         content = match.group(2)

#         # --- DEBUGGING ---
#         print(f"        --- Content for '{name}' start ---")
#         print(content)
#         print(f"        --- Content for '{name}' end ---")
#         # --- END DEBUGGING ---

#         graphql_match = graphql_pattern.search(content)
#         python_match = python_pattern.search(content)

#         if graphql_match and python_match:
#             query = graphql_match.group(1).strip()
#             # Extract the python code string, removing comments if necessary for exec
#             variables_code_str = python_match.group(1).strip()
#             # Remove comments starting with # to avoid issues with exec
#             variables_code_str = re.sub(r'^#.*$', '', variables_code_str, flags=re.MULTILINE).strip()

#             # Attempt to parse the variable assignment part more robustly if it's simple
#             try:
#                 # A simple approach might assume the last line is `variables = ...`
#                 # More robustly, find the assignment
#                 assignment_match = re.search(r'variables\s*=\s*(\{.*?\})', variables_code_str, re.DOTALL)
#                 variables_dict_code = assignment_match.group(1) if assignment_match else variables_code_str
#                 # --- DEBUGGING ---
#                 print(f"        >>> Appending example: {name}")
#                 # --- END DEBUGGING ---
#                 examples.append({
#                     "name": name,
#                     "query": query,
#                     "variables_code": variables_dict_code # Store the code string for the dict/assignment
#                 })
#             except Exception as e:
#                 print(f"Warning: Could not parse variables for example '{name}'. Error: {e}")
#                 # Decide if you want to skip or add with None/error marker
#                 # examples.append({"name": name, "query": query, "variables_code": None, "error": str(e)})

#     # --- DEBUGGING ---
#     print(f"    Finished finditer loop. Total matches found: {matches_found}")
#     print(f"<<< DEBUG: Exiting extract_gql_examples_from_docstring\n")
#     # --- END DEBUGGING ---

#     if not examples:
#          raise ValueError("No examples found in docstring. Check delimiters and file content.")

#     return examples

# # --- Pytest Fixture for Loading Examples ---
# @pytest.fixture(scope="session")
# def gql_examples():
#     """Reads the target function's docstring and extracts GQL examples."""
#     try:
#         target_docstring = inspect.getdoc(query_wandb_tool)
#         if not target_docstring:
#             raise ImportError(f"Could not get docstring for query_wandb_tool.")

#         # --- DEBUGGING: Print the retrieved docstring ---
#         print("\n--- Retrieved Docstring by inspect.getdoc() ---")
#         print(target_docstring)
#         print("--- End of Retrieved Docstring ---\n")
#         # --- END DEBUGGING ---

#         extracted = extract_gql_examples_from_docstring(target_docstring)
#         # Filter out examples where variables couldn't be parsed if the helper function indicates so
#         valid_examples = [ex for ex in extracted if ex.get("variables_code")]
#         if not valid_examples:
#              raise ValueError("No valid examples with variable code found after parsing.")
#         return valid_examples
#     except Exception as e:
#         # pytest will report this error during fixture setup
#         pytest.fail(f"Failed to setup gql_examples fixture: {e}", pytrace=False)

# _example_names = []
# try:
#     # Attempt to pre-load examples just to get names for parameterization
#     # Note: This duplicates loading but simplifies parametrize setup
#     # The fixture ensures the main test execution uses the proper setup/cached result.
#     _target_docstring = inspect.getdoc(query_wandb_tool)
#     if not _target_docstring:
#          raise ImportError("Docstring not found at collection time.")
#     _extracted_examples = extract_gql_examples_from_docstring(_target_docstring)
#     _example_names = [ex["name"] for ex in _extracted_examples if ex.get("variables_code")]
#     if not _example_names:
#          raise ValueError("No valid example names found at collection time.")
# except Exception as e:
#     print(f"Warning during test collection: Could not pre-load example names - {e}")
#     # If collection fails to get names, the test function relying on the fixture
#     # will fail later during setup/execution, which is acceptable.
#     _example_names = ["SETUP_ERROR_DURING_COLLECTION"] # Provide a placeholder


# # --- Test Function ---

# # Apply the live_api marker
# @pytest.mark.live_api
# @pytest.mark.parametrize(
#     "name", # Parametrize only by the example name
#     _example_names
# )
# def test_wandb_gql_example(name, gql_examples): # Inject fixture here, remove query/variables_code
#     """Runs a test for each extracted GraphQL example using live API calls."""

#     if name == "SETUP_ERROR_DURING_COLLECTION":
#         pytest.fail("Test collection could not determine example names. Check setup.")

#     # Find the correct example data from the fixture result based on the parameterized name
#     example_data = next((ex for ex in gql_examples if ex['name'] == name), None)
#     if not example_data:
#         pytest.fail(f"Could not find example data for name '{name}' in gql_examples fixture result.")

#     # Use the data looked up from the fixture
#     query = example_data["query"]
#     variables_code = example_data["variables_code"]

#     # The rest of the test logic remains largely the same...
#     print(f"\nRunning test for example: {name}")
#     print(f"Query:\n{query}")
#     print(f"Variables Code:\n{variables_code}")

#     variables = {}
#     try:
#         # Execute the Python code string to get the variables dictionary.
#         # Reverting to exec as ast.literal_eval cannot handle nested strings required for JSON literals.
#         local_scope = {'json': json} # Provide json module in the execution scope
#         # The variable `variables_code` should contain the raw python code from the docstring block
#         exec(variables_code, local_scope)

#         # Check if 'variables' was defined in the executed code
#         if 'variables' not in local_scope:
#             raise NameError("Executed code snippet did not define a 'variables' dictionary.")

#         variables = local_scope['variables']

#         if not isinstance(variables, dict):
#              raise TypeError(f"Executed code defined 'variables', but it is not a dictionary. Got: {type(variables)}")

#         print(f"Original Variables: {variables}")

#         # Override entity and project for the test run
#         # Check if the keys exist before assigning, especially for mutations
#         if 'entity' in variables or name.endswith('Info') or name.endswith('Runs') or name.endswith('Keys') or name.endswith('Sampled') or name.endswith('Details'):
#             variables['entity'] = TARGET_ENTITY
#         if 'project' in variables or name.endswith('Info') or name.endswith('Runs') or name.endswith('Keys') or name.endswith('Sampled') or name.endswith('Details'):
#             variables['project'] = TARGET_PROJECT
#         # Handle entityName/projectName variants if needed
#         if 'entityName' in variables:
#              variables['entityName'] = TARGET_ENTITY
#         if 'projectName' in variables:
#              variables['projectName'] = TARGET_PROJECT

#         # Specific override for GetArtifactDetails test
#         if name == 'GetArtifactDetails':
#              # Use the specific artifact name provided by the user
#              variables['artifactName'] = "c-metrics/hallucination/SmolLM2-360M-sft-hallu:v12"
#              print(f"    Overriding artifactName for {name} test.") # Debug print

#         # Handle mutations which might not have standard entity/project vars
#         if name == 'UpsertProject' or name == 'CreateProject':
#              # Ensure the mutation targets the test entity, adjust name if needed
#              variables['entity'] = TARGET_ENTITY
#              variables['name'] = f"{TARGET_PROJECT}-test-upsert" # Avoid conflicts


#         # Handle cases where limit might be needed but not in example vars (like mutations)
#         # For mutations, the tool itself might not use max_items, depends on implementation
#         # For queries, ensure a reasonable limit if not present? Or rely on tool default.
#         # Let's rely on the tool's default `max_items` for now.

#         print(f"Modified Variables: {variables}")


#     except Exception as e:
#         pytest.fail(f"Failed to execute or modify variables code for example '{name}': {e}\nCode: {variables_code}")

#     # --- Make the Live API Call ---
#     try:
#         # Use default max_items and items_per_page from the tool's signature
#         result = query_wandb_tool(query=query, variables=variables)

#         print(f"API Result for {name}: {result}")

#         # --- Assertions ---
#         assert isinstance(result, dict), f"Expected result to be a dictionary, got {type(result)}"

#         # Check specifically for the 'errors' key which indicates GraphQL level errors
#         if 'errors' in result:
#              # Sometimes 'errors' is present but None or empty list, check content
#              error_content = result.get('errors')
#              assert not error_content, f"GraphQL API returned errors for example '{name}': {error_content}"

#         # Optional: Add more specific checks based on the query name if needed
#         # e.g., if name == "GetProjectInfo": assert "project" in result.get("data", {})

#     except Exception as e:
#         pytest.fail(f"query_wandb_tool raised an exception for example '{name}': {e}")

# # Note: This test makes live calls to the W&B API. Ensure:
# # 1. You are logged into W&B (e.g., via `wandb login`).
# # 2. The target project (c-metrics/hallucination) exists and is accessible.
# # 3. Network connectivity is available.
# # 4. Be mindful of API rate limits if running frequently.
# # To run only these tests: pytest -m live_api
# # To skip these tests: pytest -m "not live_api"
