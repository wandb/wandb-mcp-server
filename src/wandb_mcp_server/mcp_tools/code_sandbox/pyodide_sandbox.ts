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
          // Will be captured by our execute method
        },
        stderr: (text: string) => {
          // Will be captured by our execute method
        },
      });

      // Load commonly used packages
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
      // Set up output capture
      const outputLines: string[] = [];
      const errorLines: string[] = [];
      
      this.pyodide.stdout = (text: string) => {
        outputLines.push(text);
      };
      
      this.pyodide.stderr = (text: string) => {
        errorLines.push(text);
      };

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

      // Execute the Python code
      const timeout = request.timeout || 30;
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), timeout * 1000);

      try {
        await this.pyodide.runPythonAsync(request.code!);
        result.success = true;
      } catch (error: any) {
        result.error = error.toString();
        if (error.type === "PythonError") {
          // Format Python traceback nicely
          result.error = this.pyodide.formatException(error);
        }
      } finally {
        clearTimeout(timeoutId);
      }

      // Capture output
      result.output = outputLines.join("");
      if (errorLines.length > 0) {
        result.logs.push(...errorLines);
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
      if (done) break;
      
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
                throw new Error("No code provided for execution");
              }
              const result = await globalSandbox.execute(request);
              console.log(JSON.stringify(result));
            }
          } catch (error) {
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
      console.error(`Server error: ${error}`);
      break;
    }
  }
}

export { PyodideSandbox, type ExecutionRequest, type ExecutionResult };