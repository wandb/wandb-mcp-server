# Test cases for the Weave MCP

## Traces
    by default hide "weave" attributes

    ### Data retrieval
    - test for op name containing "VectorStore.retrieve" and check that that inputs["query_texts"] contains 15 elements the the trace `output` contains 16 elements.
    - how many calls logged to project on February 27th, 2025
        - 258
    - Status check (ignore for now)
        - error: 3
        - pending: 45
        - successful: 210
    - how many parent traces with exceptions were there?
        - Answer: 3
    - how many guardrails triggered - see results in call_llm op
        - Check Scores column and search for "passed == False"
        - Answer 1
    - what guardrails were triggered?
        - Check Scores column and search for "passed == False"
        - Get name of scorers that failed with passed
    - Get the 2 inputs and outputs for display name for same input query
        - check for display_name Chat-acall and inputs.chat_request.question == "Can I download a model from W&B artifact without a W&B API key?"
    - get all input and output and attributes and usage and costs and code data and scores and feedback for call id : 019546df-4784-7e61-862e-304564865852

    - how did the openai system prompt evolve in the "how_to_catch_a_pirate" app?
         - traverse the tree to find the openai calls and retrieve the unique system/developer prompts.
         - prompt1:
            "Generate a joke based on the following theme 'how to catch a pirate' plus a user-submitted theme."
         - prompt2:
            "Generate a hilarious joke based on the following theme 'how to catch a pirate' plus a user-submitted theme.make it wildly creative and artistic. take inspiration from 1980s comedians." 
    - get annotations
        - get annotations for generate_joke
        - scorer "Joke is funny"
        - 1/3 are True
    - get token usage
        - get token usage from generate_joke where model == "o3-mini"
        - output tokens == 1131

    - get costs
        return all costs from generate_joke

    - get attrributes

    - get preview of data...

    - summary calls for previews ....

    - All inputs and outputs for display name
        - display_name == Chat-acall
        - inputs.chat_request.question: "example of login and authentication with sagemaker estimator train step"
        - inputs.chat_request.language: "en"
        - len(outputs.response_synthesis_llm_messages) == 6
        - outputs.start_time = datetime.datetime(2025, 2, 27, 10, 6, 32, 836545, tzinfo=datetime.timezone.utc)
        - outputs.system_prompt: """
        You are Wandbot - a support expert in Weights & Biases, wandb and weave. 
Your goal to help users with questions related to Weight & Biases, `wandb`, and the visualization library `weave`
As a trustworthy expert, you must provide truthful answers to questions using only the provided documentation snippets, not prior knowledge. 
Here are guidelines you must follow when responding to user questions:

**Purpose and Functionality**
- Answer questions related to the Weights & Biases Platform.
- Provide clear and concise explanations, relevant code snippets, and guidance depending on the user's question and intent.
- Ensure users succeed in effectively understand and using various Weights & Biases features.
- Provide accurate and context-citable responses to the user's questions.

**Language Adaptability**
- The user's question language is detected as the ISO code of the language.
- Always respond in the detected question language.

**Specificity**
- Be specific and provide details only when required.
- Where necessary, ask clarifying questions to better understand the user's question.
- Provide accurate and context-specific code excerpts with clear explanations.
- Ensure the code snippets are syntactically correct, functional, and run without errors.
- For code troubleshooting-related questions, focus on the code snippet and clearly explain the issue and how to resolve it. 
- Avoid boilerplate code such as imports, installs, etc.

**Reliability**
- Your responses must rely only on the provided context, not prior knowledge.
- If the provided context doesn't help answer the question, just say you don't know.
- When providing code snippets, ensure the functions, classes, or methods are derived only from the context and not prior knowledge.
- Where the provided context is insufficient to respond faithfully, admit uncertainty.
- Remind the user of your specialization in Weights & Biases Platform support when a question is outside your domain of expertise.
- Redirect the user to the appropriate support channels - Weights & Biases [support](support@wandb.com) or [community forums](https://wandb.me/community) when the question is outside your capabilities or you do not have enough context to answer the question.

**Citation**
- Always cite the source from the provided context.
- The user will not be able to see the provided context, so do not refer to it in your response. For instance, don't say "As mentioned in the context...".
- Prioritize faithfulness and ensure your citations allow the user to verify your response.
- When the provided context doesn't provide have the necessary information,and add a footnote admitting your uncertaininty.
- Remember, you must return both an answer and citations.


**Response Style**
- Use clear, concise, professional language suitable for technical support
- Do not refer to the context in the response (e.g., "As mentioned in the context...") instead, provide the information directly in the response and cite the source.


**Response Formatting**
- Always communicate with the user in Markdown.
- Do not use headers in your output as it will be rendered in slack.
- Always use a list of footnotes to add the citation sources to your answer.

**Example**:

The correct answer to the user's query

 Steps to solve the problem:
 - **Step 1**: ...[^1], [^2]
 - **Step 2**: ...[^1]
 ...

 Here's a code snippet[^3]

 ```python
 # Code example
 ...
 ```
 
 **Explanation**:

 - Point 1[^2]
 - Point 2[^3]

 **Sources**:

 - [^1]: [source](source_url)
 - [^2]: [source](source_url)
 - [^3]: [source](source_url)
 ...
        """


    ### Write data
    - add feedback

    ### Data stats
    - how many traces in the project?
    - trace counts by name?

## Evaluations
    - how many trials
    - Look at failed examples
        - count them
        - identify common errors
    - get the F1 score for the last 10 results
    - get the precision for the eval called XX

## Datasets
    - query size and stats
    - Is there a sample like xxx in my dataset
    - add to dataset

# TODOs
## Images
## Prompts
    - ask about prompts
    - push new prompt?
    - "attach from MCP" - pull in prompts

# Sandbox Tests

## test_sandbox_execution.py (Unit Tests)
Unit tests using mocking to test sandbox logic without requiring actual sandboxes:
- **ExecutionCache**: Tests LRU cache operations, eviction, and TTL expiration
- **RateLimiter**: Tests rate limiting with window expiration
- **E2BSandboxPool**: Tests pool initialization and acquire/release (now simplified to single instance)
- **E2BSandbox**: Tests package installation, code execution via file write
- **PyodideSandbox**: Tests Deno availability checking and execution
- **Main execution function**: Tests rate limiting, caching, fallback behavior

## test_sandbox_integration.py (Integration Tests)
Integration tests that execute code in real sandboxes:
- **Basic execution**: Tests code execution across all available sandboxes
- **E2B features**: Tests package installation, file operations, subprocess support
- **Pyodide features**: Tests WebAssembly isolation, standard library support
- **Error handling**: Tests timeout handling, error propagation
- **Concurrent execution**: Tests multiple simultaneous executions
- **Data science workflow**: Tests realistic analysis scenarios

## test_sandbox_persistence.py (File Persistence Tests)
Tests for file persistence and session management:
- **E2B persistence**: Tests that files persist across multiple executions
- **E2B instance reuse**: Verifies the same sandbox instance is used
- **Pyodide persistence**: Tests that files persist across multiple executions in the same process
- **Sandbox file utilities**: Tests the write_json_to_sandbox function
- **Native file operations**: Tests E2B's files.write() and Pyodide's writeFile()
- **Concurrent file operations**: Tests multiple file writes
- **W&B integration**: Tests writing query results to sandbox files

## test_sandbox_file_operations.py (File Operations Tests)
Tests for file system operations in sandboxes:
- **Native file operations**: Direct file write/read using sandbox APIs
- **Binary file handling**: Tests base64 encoding for binary data
- **Large file handling**: Tests with MB-sized files
- **File permissions**: Tests error handling for missing files, directories
- **Directory operations**: Tests creating and listing directory structures
- **CSV processing**: Tests data analysis workflows with CSV files
- **Security isolation**: Tests filesystem and network isolation

## test_sandbox_mcp_tool.py (MCP Tool Tests)
Tests for the MCP sandbox tool integration:
- **Tool schema generation**: Tests Anthropic tool schema format
- **Successful execution**: Tests normal code execution flow
- **Error handling**: Tests syntax errors and exceptions
- **Parameter handling**: Tests timeout, sandbox_type, install_packages
- **Anthropic integration**: Tests with actual Anthropic API (when API key available)

## Running Sandbox Tests

```bash
# Run all sandbox tests
uv run pytest tests/test_sandbox*.py -v

# Run specific test category
uv run pytest tests/test_sandbox_persistence.py -v

# Run with debug logging
MCP_SERVER_LOG_LEVEL=DEBUG uv run pytest tests/test_sandbox*.py -v -s
```

Requirements:
- **E2B tests**: Require `E2B_API_KEY` environment variable
- **Pyodide tests**: Require Deno installed (`curl -fsSL https://deno.land/install.sh | sh`)