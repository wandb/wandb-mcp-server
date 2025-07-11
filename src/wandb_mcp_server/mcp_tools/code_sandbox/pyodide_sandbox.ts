#!/usr/bin/env -S deno run --allow-net --allow-read --allow-write --allow-env

/**
 * Direct Pyodide sandbox implementation for Python code execution.
 * 
 * Based on Pydantic AI's mcp-run-python server:
 * https://github.com/pydantic/pydantic-ai/tree/main/mcp-run-python
 * 
 * We've adapted their approach of using Deno + Pyodide for sandboxed Python execution
 * with our own implementation that provides direct file system access and better
 * integration with our MCP server architecture.
 */

import { loadPyodide } from "npm:pyodide@0.26.4";

interface ExecutionRequest {
  type?: "execute" | "writeFile";
  code?: string;
  path?: string;
  content?: string;
  files?: { [path: string]: string };
  timeout?: number;
}

interface ExecutionResult {
  success: boolean;
  output: string;
  error: string | null;
  logs: string[];
}

class PyodideSandbox {
  private pyodide: any = null;
  private initialized = false;

  async initialize() {
    if (this.initialized) return;

    try {
      console.error("Initializing Pyodide...");
      this.pyodide = await loadPyodide({
        stdout: (text: string) => {
          // Redirect to stderr during initialization to avoid interfering with JSON responses
          console.error(`Pyodide stdout: ${text}`);
        },
        stderr: (text: string) => {
          // Redirect to stderr during initialization
          console.error(`Pyodide stderr: ${text}`);
        },
      });

      // Set up interrupt buffer for cancelling long-running code
      // This allows us to interrupt Python execution
      this.pyodide.setInterruptBuffer(new Uint8Array(new SharedArrayBuffer(4)));
      
      // Load commonly used packages - output will go to stderr
      console.error("Loading packages: numpy, pandas, matplotlib...");
      await this.pyodide.loadPackage(["numpy", "pandas", "matplotlib"]);
      
      this.initialized = true;
      console.error("Pyodide initialized successfully");
    } catch (error) {
      console.error("Failed to initialize Pyodide:", error);
      throw error;
    }
  }

  async execute(request: ExecutionRequest): Promise<ExecutionResult> {
    if (!this.initialized) {
      await this.initialize();
    }

    const result: ExecutionResult = {
      success: false,
      output: "",
      error: null,
      logs: [],
    };

    try {
      // Set up output capture BEFORE running any Python code
      
      // First, set up Python's sys.stdout and sys.stderr to capture output
      await this.pyodide.runPythonAsync(`
import sys
from io import StringIO

# Create StringIO objects to capture output
_stdout_buffer = StringIO()
_stderr_buffer = StringIO()

# Save original stdout/stderr
_original_stdout = sys.stdout
_original_stderr = sys.stderr

# Redirect stdout/stderr
sys.stdout = _stdout_buffer
sys.stderr = _stderr_buffer
      `);

      // Write any input files to the virtual filesystem
      if (request.files) {
        for (const [path, content] of Object.entries(request.files)) {
          try {
            // Ensure directory exists
            const dir = path.substring(0, path.lastIndexOf('/'));
            if (dir) {
              this.pyodide.FS.mkdirTree(dir);
            }
            
            // Write file
            this.pyodide.FS.writeFile(path, content);
            result.logs.push(`Wrote file: ${path}`);
          } catch (error) {
            result.logs.push(`Failed to write file ${path}: ${error}`);
          }
        }
      }

      // Execute the Python code with timeout
      const timeout = request.timeout || 30;
      
      // Set up timeout handling with interrupt
      let timeoutId: number | null = null;
      let interrupted = false;
      
      const timeoutPromise = new Promise((_, reject) => {
        timeoutId = setTimeout(() => {
          // Interrupt the Python execution
          try {
            this.pyodide.interruptExecution();
            interrupted = true;
          } catch (e) {
            console.error("Failed to interrupt execution:", e);
          }
          reject(new Error(`Execution timed out after ${timeout} seconds`));
        }, timeout * 1000);
      });

      try {
        // Race between execution and timeout
        const executionResult = await Promise.race([
          this.pyodide.runPythonAsync(request.code!),
          timeoutPromise
        ]);
        
        // Clear timeout if execution completed
        if (timeoutId !== null) {
          clearTimeout(timeoutId);
        }
        
        // Capture the output from the StringIO buffers
        const capturedOutput = await this.pyodide.runPythonAsync(`
# Get captured output
_stdout_output = _stdout_buffer.getvalue()
_stderr_output = _stderr_buffer.getvalue()

# Restore original stdout/stderr
sys.stdout = _original_stdout
sys.stderr = _original_stderr

# Clean up
_stdout_buffer.close()
_stderr_buffer.close()

# Return the captured output
(_stdout_output, _stderr_output)
        `);
        
        const [stdoutOutput, stderrOutput] = capturedOutput.toJs();
        
        result.success = true;
        result.output = stdoutOutput || "";
        
        // Add the return value if it's not None
        if (executionResult !== undefined && executionResult !== null && executionResult !== this.pyodide.globals.get('None')) {
          if (result.output && !result.output.endsWith('\n')) {
            result.output += "\n";
          }
          result.output += String(executionResult);
        }
        
        if (stderrOutput) {
          result.logs.push(stderrOutput);
        }
      } catch (error: any) {
        if (interrupted || error.message?.includes("timed out")) {
          // This is a timeout/interrupt error
          result.error = `Execution timed out after ${timeout} seconds`;
          result.success = false;
          
          // Try to capture any partial output
          try {
            const capturedOutput = await this.pyodide.runPythonAsync(`
# Get captured output even if there was a timeout
_stdout_output = _stdout_buffer.getvalue() if '_stdout_buffer' in locals() else ""
_stderr_output = _stderr_buffer.getvalue() if '_stderr_buffer' in locals() else ""

# Restore original stdout/stderr if they exist
if '_original_stdout' in locals():
    sys.stdout = _original_stdout
if '_original_stderr' in locals():
    sys.stderr = _original_stderr

# Clean up if buffers exist
if '_stdout_buffer' in locals():
    _stdout_buffer.close()
if '_stderr_buffer' in locals():
    _stderr_buffer.close()

(_stdout_output, _stderr_output)
            `);
            
            const [stdoutOutput, stderrOutput] = capturedOutput.toJs();
            if (stdoutOutput) {
              result.output = stdoutOutput;
            }
            if (stderrOutput) {
              result.logs.push(stderrOutput);
            }
          } catch {
            // Ignore errors in cleanup
          }
          
          return result;
        }
        
        // Regular error (not timeout)
        // Try to capture any output before the error
        try {
          const capturedOutput = await this.pyodide.runPythonAsync(`
# Get captured output even if there was an error
_stdout_output = _stdout_buffer.getvalue() if '_stdout_buffer' in locals() else ""
_stderr_output = _stderr_buffer.getvalue() if '_stderr_buffer' in locals() else ""

# Restore original stdout/stderr if they exist
if '_original_stdout' in locals():
    sys.stdout = _original_stdout
if '_original_stderr' in locals():
    sys.stderr = _original_stderr

# Clean up if buffers exist
if '_stdout_buffer' in locals():
    _stdout_buffer.close()
if '_stderr_buffer' in locals():
    _stderr_buffer.close()

(_stdout_output, _stderr_output)
          `);
          
          const [stdoutOutput, stderrOutput] = capturedOutput.toJs();
          if (stdoutOutput) {
            result.output = stdoutOutput;
          }
          if (stderrOutput) {
            result.logs.push(stderrOutput);
          }
        } catch {
          // Ignore errors in cleanup
        }
        
        // Handle Python errors specially
        if (error && error.constructor && error.constructor.name === "PythonError") {
          // Pyodide wraps Python exceptions - extract the actual error type
          const errorStr = error.toString();
          const errorMessage = error.message || errorStr;
          
          // Try to extract the Python error type from the traceback
          // Python tracebacks typically show the error type at the end
          const syntaxErrorMatch = errorMessage.match(/SyntaxError:|IndentationError:|NameError:|TypeError:|ValueError:|AttributeError:|KeyError:|IndexError:|ZeroDivisionError:/);
          if (syntaxErrorMatch) {
            result.error = errorMessage; // Include full traceback
          } else {
            // Fallback - at least mention it's a Python error
            result.error = `PythonError: ${errorMessage}`;
          }
        } else if (error && error.message) {
          result.error = error.message;
        } else {
          result.error = error.toString();
        }
      }

    } catch (error: any) {
      result.error = `Sandbox execution failed: ${error.toString()}`;
    }

    return result;
  }

  /**
   * Write a file directly to the Pyodide filesystem.
   * This is more efficient than executing Python code to write files.
   */
  async writeFile(path: string, content: string): Promise<void> {
    if (!this.initialized) {
      await this.initialize();
    }

    try {
      // Ensure directory exists
      const dir = path.substring(0, path.lastIndexOf('/'));
      if (dir) {
        this.pyodide.FS.mkdirTree(dir);
      }
      
      // Write file
      this.pyodide.FS.writeFile(path, content);
    } catch (error) {
      throw new Error(`Failed to write file ${path}: ${error}`);
    }
  }

  /**
   * Read a file from the Pyodide filesystem.
   */
  async readFile(path: string): Promise<string> {
    if (!this.initialized) {
      await this.initialize();
    }

    try {
      return this.pyodide.FS.readFile(path, { encoding: "utf8" });
    } catch (error) {
      throw new Error(`Failed to read file ${path}: ${error}`);
    }
  }
}

// Global persistent sandbox instance
let globalSandbox: PyodideSandbox | null = null;

// Main execution when called directly
// deno-lint-ignore-file no-explicit-any
if (import.meta.main) {
  // Initialize sandbox once on startup
  if (!globalSandbox) {
    console.error("Starting persistent Pyodide sandbox server...");
    globalSandbox = new PyodideSandbox();
    await globalSandbox.initialize();
    console.error("Pyodide sandbox server ready");
  }
  
  // Read commands from stdin in a loop
  const decoder = new TextDecoder();
  const reader = Deno.stdin.readable.getReader();
  
  while (true) {
    try {
      const { value, done } = await reader.read();
      if (done) {
        console.error("Stdin closed, exiting...");
        break;
      }
      
      if (value) {
        const lines = decoder.decode(value).trim().split('\n');
        for (const line of lines) {
          if (!line) continue;
          
          try {
            const request: ExecutionRequest = JSON.parse(line);
            
            // Handle different request types
            if (request.type === "writeFile" && request.path && request.content !== undefined) {
              // Handle file write request
              try {
                await globalSandbox.writeFile(request.path, request.content);
                const writeResult: ExecutionResult = {
                  success: true,
                  output: `File written to ${request.path}`,
                  error: null,
                  logs: [],
                };
                console.log(JSON.stringify(writeResult));
              } catch (error) {
                const errorResult: ExecutionResult = {
                  success: false,
                  output: "",
                  error: `Failed to write file: ${error}`,
                  logs: [],
                };
                console.log(JSON.stringify(errorResult));
              }
            } else {
              // Default to code execution (backward compatibility)
              if (!request.code) {
                const errorResult: ExecutionResult = {
                  success: false,
                  output: "",
                  error: "No code provided for execution",
                  logs: [],
                };
                console.log(JSON.stringify(errorResult));
                continue;
              }
              
              try {
                const result = await globalSandbox.execute(request);
                console.log(JSON.stringify(result));
              } catch (error) {
                const errorResult: ExecutionResult = {
                  success: false,
                  output: "",
                  error: `Execution failed: ${error}`,
                  logs: [],
                };
                console.log(JSON.stringify(errorResult));
              }
            }
          } catch (error) {
            // JSON parsing or other request processing errors
            const errorResult: ExecutionResult = {
              success: false,
              output: "",
              error: `Failed to process request: ${error}`,
              logs: [],
            };
            console.log(JSON.stringify(errorResult));
          }
        }
      }
    } catch (error) {
      // Log the error but don't break the loop unless it's a critical error
      console.error(`Server error: ${error}`);
      
      // Only break on critical errors that indicate the process should exit
      if (error instanceof Deno.errors.BrokenPipe || 
          error instanceof Deno.errors.ConnectionReset ||
          error.name === "BadResource") {
        console.error("Critical error detected, exiting...");
        break;
      }
      
      // For other errors, send an error response and continue
      try {
        const errorResult: ExecutionResult = {
          success: false,
          output: "",
          error: `Server error: ${error}`,
          logs: [],
        };
        console.log(JSON.stringify(errorResult));
      } catch (outputError) {
        console.error(`Failed to send error response: ${outputError}`);
        // If we can't even send an error response, the connection is likely broken
        break;
      }
    }
  }
  
  console.error("Pyodide sandbox server shutting down");
}

export { PyodideSandbox, type ExecutionRequest, type ExecutionResult };