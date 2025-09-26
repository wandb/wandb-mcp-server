#!/usr/bin/env python3
"""
Quick script to check if the W&B MCP Server is running correctly.

Usage:
    python check_server.py [--url SERVER_URL] [--port PORT]
"""

import os
import sys
import argparse
import requests
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def check_server(base_url="http://localhost:7860", api_key=None):
    """Check server endpoints and health."""
    
    print("🔍 Checking W&B MCP Server Status")
    print("=" * 50)
    print(f"Server: {base_url}")
    
    # Check health endpoint
    print("\n1. Health Check:")
    try:
        response = requests.get(f"{base_url}/health", timeout=2)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Server is healthy")
            print(f"   - Service: {data.get('service', 'unknown')}")
            print(f"   - Status: {data.get('status', 'unknown')}")
            if 'version' in data:
                print(f"   - Version: {data.get('version')}")
        elif response.status_code == 404:
            print(f"   ❌ Health endpoint not found (404)")
            print(f"   Server may not be configured correctly")
            return False
        else:
            print(f"   ❌ Unexpected status: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"   ❌ Server not running at {base_url}")
        print(f"   Start with: uv run app.py")
        return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    
    # Check MCP without auth (should fail)
    print("\n2. MCP Authentication Check:")
    try:
        response = requests.post(
            f"{base_url}/mcp",
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json, text/event-stream"
            },
            json={"jsonrpc": "2.0", "method": "initialize", "params": {}, "id": 1},
            timeout=2
        )
        if response.status_code == 401:
            print(f"   ✅ Correctly requires authentication (401)")
        elif response.status_code == 406:
            print(f"   ⚠️  Server expects different Accept header")
        else:
            print(f"   ⚠️  Unexpected status without auth: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Could not check MCP endpoint: {e}")
        return False
    
    # If API key provided, test with auth
    if api_key:
        print("\n3. MCP Authentication Test:")
        try:
            response = requests.post(
                f"{base_url}/mcp",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream"
                },
                json={
                    "jsonrpc": "2.0",
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "1.0.0",
                        "capabilities": {},
                        "clientInfo": {"name": "check_server", "version": "1.0"}
                    },
                    "id": 2
                },
                timeout=5
            )
            if response.status_code == 200:
                session_id = response.headers.get("mcp-session-id")
                if session_id:
                    print(f"   ✅ Authentication successful")
                    print(f"   - Session ID: {session_id[:20]}...")
                else:
                    print(f"   ⚠️  No session ID returned")
            else:
                print(f"   ❌ Authentication failed: {response.status_code}")
                if response.text:
                    print(f"   - Response: {response.text[:200]}")
        except Exception as e:
            print(f"   ❌ Error testing authentication: {e}")
    
    print("\n" + "=" * 50)
    print("✅ Server health check complete")
    
    if not api_key:
        print("\n💡 Tip: Set WANDB_API_KEY or use --api-key to test authentication")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Check W&B MCP Server health')
    parser.add_argument('--url', default=None, help='Server URL (default: http://localhost:7860)')
    parser.add_argument('--port', type=int, default=None, help='Server port (overrides URL)')
    parser.add_argument('--api-key', default=None, help='W&B API key for auth test')
    
    args = parser.parse_args()
    
    # Determine server URL
    if args.url:
        base_url = args.url
    elif args.port:
        base_url = f"http://localhost:{args.port}"
    else:
        base_url = os.environ.get("MCP_TEST_SERVER_URL", "http://localhost:7860")
    
    # Get API key
    api_key = args.api_key or os.environ.get("WANDB_API_KEY")
    
    success = check_server(base_url, api_key)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

