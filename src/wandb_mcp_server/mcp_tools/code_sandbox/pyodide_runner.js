#!/usr/bin/env node
/**
 * Pyodide Runner - Executes Python code in a WebAssembly sandbox using Pyodide
 * 
 * Usage: node pyodide_runner.js <base64_encoded_code>
 * 
 * Output format (JSON):
 * {
 *   "success": boolean,
 *   "output": string,
 *   "error": string | null,
 *   "execution_time_ms": number
 * }
 */

const fs = require('fs');
const path = require('path');

// Check if pyodide is available locally or needs to be downloaded
let pyodideModule;
try {
    pyodideModule = require('pyodide');
} catch (e) {
    // If pyodide is not installed locally, we'll use the CDN version
    pyodideModule = null;
}

async function runPythonCode(code) {
    const startTime = Date.now();
    let pyodide;
    
    try {
        // Load Pyodide
        if (pyodideModule) {
            pyodide = await pyodideModule.loadPyodide();
        } else {
            // Use dynamic import for CDN version
            const { loadPyodide } = await import('https://cdn.jsdelivr.net/pyodide/v0.24.1/full/pyodide.mjs');
            pyodide = await loadPyodide({
                indexURL: "https://cdn.jsdelivr.net/pyodide/v0.24.1/full/"
            });
        }
        
        // Pre-load commonly used packages
        try {
            await pyodide.loadPackage(["numpy", "pandas", "matplotlib", "scipy"]);
        } catch (e) {
            // Some packages might not be available, continue anyway
            console.error("Warning: Could not load some packages:", e.message);
        }
        
        // Set up Python environment to capture output
        pyodide.runPython(`
import sys
from io import StringIO
import traceback

# Create output capture
_stdout_capture = StringIO()
_stderr_capture = StringIO()
_original_stdout = sys.stdout
_original_stderr = sys.stderr

# Redirect stdout and stderr
sys.stdout = _stdout_capture
sys.stderr = _stderr_capture

# Store the result
_execution_result = {
    'success': False,
    'output': '',
    'error': None
}

try:
    # Execute the user code
    exec('''${code.replace(/'/g, "\\'")}''')
    
    # Capture output
    _execution_result['output'] = _stdout_capture.getvalue()
    _execution_result['success'] = True
    
except Exception as e:
    # Capture error with full traceback
    _execution_result['error'] = traceback.format_exc()
    _execution_result['output'] = _stdout_capture.getvalue()
    
finally:
    # Restore original stdout/stderr
    sys.stdout = _original_stdout
    sys.stderr = _original_stderr
`);
        
        // Get the execution result
        const result = pyodide.globals.get('_execution_result').toJs();
        
        // Clean up
        pyodide.globals.delete('_execution_result');
        pyodide.globals.delete('_stdout_capture');
        pyodide.globals.delete('_stderr_capture');
        
        return {
            success: result.get('success'),
            output: result.get('output') || '',
            error: result.get('error'),
            execution_time_ms: Date.now() - startTime
        };
        
    } catch (error) {
        return {
            success: false,
            output: '',
            error: `Pyodide initialization or execution failed: ${error.message}`,
            execution_time_ms: Date.now() - startTime
        };
    } finally {
        // Clean up Pyodide instance if it exists
        if (pyodide) {
            try {
                // Pyodide doesn't have a direct cleanup method, but we can clear globals
                pyodide.runPython('globals().clear()');
            } catch (e) {
                // Ignore cleanup errors
            }
        }
    }
}

// Main execution
async function main() {
    if (process.argv.length < 3) {
        console.error(JSON.stringify({
            success: false,
            output: '',
            error: 'Usage: node pyodide_runner.js <base64_encoded_code>',
            execution_time_ms: 0
        }));
        process.exit(1);
    }
    
    try {
        // Decode the base64 encoded Python code
        const encodedCode = process.argv[2];
        const code = Buffer.from(encodedCode, 'base64').toString('utf-8');
        
        // Run the code
        const result = await runPythonCode(code);
        
        // Output the result as JSON
        console.log(JSON.stringify(result));
        
        process.exit(result.success ? 0 : 1);
        
    } catch (error) {
        console.error(JSON.stringify({
            success: false,
            output: '',
            error: `Failed to process input: ${error.message}`,
            execution_time_ms: 0
        }));
        process.exit(1);
    }
}

// Handle unhandled promise rejections
process.on('unhandledRejection', (reason, promise) => {
    console.error(JSON.stringify({
        success: false,
        output: '',
        error: `Unhandled promise rejection: ${reason}`,
        execution_time_ms: 0
    }));
    process.exit(1);
});

// Run the main function
main().catch(error => {
    console.error(JSON.stringify({
        success: false,
        output: '',
        error: `Fatal error: ${error.message}`,
        execution_time_ms: 0
    }));
    process.exit(1);
});