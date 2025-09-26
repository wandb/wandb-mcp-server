#!/usr/bin/env python3
"""
Load testing script for W&B MCP Server
Measures concurrent connections, requests/second, and latency
"""

import asyncio
import time
import statistics
from typing import List, Dict, Any, Optional
import httpx
import json
from datetime import datetime
import argparse
import sys

class MCPLoadTester:
    def __init__(self, base_url: str = "http://localhost:7860", api_key: str = None):
        self.base_url = base_url
        self.api_key = api_key or "test_key_12345678901234567890123456789012345678"
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "response_times": [],
            "session_creation_times": [],
            "tool_call_times": []
        }
    
    async def create_session(self, client: httpx.AsyncClient) -> Optional[str]:
        """Initialize an MCP session."""
        start_time = time.time()
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        
        payload = {
            "jsonrpc": "2.0",
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "load_test", "version": "1.0.0"}
            },
            "id": 1
        }
        
        try:
            response = await client.post(
                f"{self.base_url}/mcp",
                headers=headers,
                json=payload,
                timeout=10
            )
            
            elapsed = time.time() - start_time
            self.metrics["session_creation_times"].append(elapsed)
            self.metrics["total_requests"] += 1
            
            if response.status_code == 200:
                self.metrics["successful_requests"] += 1
                return response.headers.get("mcp-session-id")
            else:
                self.metrics["failed_requests"] += 1
                return None
                
        except Exception as e:
            self.metrics["failed_requests"] += 1
            self.metrics["total_requests"] += 1
            print(f"Session creation failed: {e}")
            return None
    
    async def call_tool(self, client: httpx.AsyncClient, session_id: str, tool_name: str, params: Dict[str, Any]):
        """Call a tool using the session."""
        start_time = time.time()
        
        headers = {
            "Mcp-Session-Id": session_id,
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        
        payload = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": params
            },
            "id": 2
        }
        
        try:
            response = await client.post(
                f"{self.base_url}/mcp",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            elapsed = time.time() - start_time
            self.metrics["tool_call_times"].append(elapsed)
            self.metrics["response_times"].append(elapsed)
            self.metrics["total_requests"] += 1
            
            if response.status_code == 200:
                self.metrics["successful_requests"] += 1
            else:
                self.metrics["failed_requests"] += 1
                
        except Exception as e:
            self.metrics["failed_requests"] += 1
            self.metrics["total_requests"] += 1
            print(f"Tool call failed: {e}")
    
    async def run_client_session(self, client_id: int, num_requests: int, delay: float = 0.1):
        """Simulate a client making multiple requests."""
        async with httpx.AsyncClient() as client:
            # Create session
            session_id = await self.create_session(client)
            if not session_id:
                return
            
            # Make multiple tool calls
            for i in range(num_requests):
                await self.call_tool(
                    client,
                    session_id,
                    "query_wandb_entity_projects",  # Simple tool that doesn't need entity/project
                    {}
                )
                
                # Small delay between requests
                if delay > 0:
                    await asyncio.sleep(delay)
    
    async def run_load_test(self, num_clients: int, requests_per_client: int, delay: float = 0.1):
        """Run the load test with specified parameters."""
        print(f"\n{'='*60}")
        print(f"Starting Load Test")
        print(f"{'='*60}")
        print(f"Clients: {num_clients}")
        print(f"Requests per client: {requests_per_client}")
        print(f"Total requests: {num_clients * (requests_per_client + 1)}")  # +1 for session creation
        print(f"Server: {self.base_url}")
        print(f"Delay between requests: {delay}s")
        print(f"{'='*60}\n")
        
        # Reset metrics
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "response_times": [],
            "session_creation_times": [],
            "tool_call_times": []
        }
        
        start_time = time.time()
        
        # Run all client sessions concurrently
        tasks = [
            self.run_client_session(i, requests_per_client, delay)
            for i in range(num_clients)
        ]
        
        # Show progress
        print("Running load test...")
        await asyncio.gather(*tasks)
        
        total_time = time.time() - start_time
        
        # Calculate and display results
        self.display_results(total_time, num_clients, requests_per_client)
        
        return self.metrics
    
    def display_results(self, total_time: float, num_clients: int, requests_per_client: int):
        """Display load test results."""
        print(f"\n{'='*60}")
        print(f"Load Test Results")
        print(f"{'='*60}")
        
        # Overall metrics
        total_requests = self.metrics["total_requests"]
        success_rate = (self.metrics["successful_requests"] / total_requests * 100) if total_requests > 0 else 0
        
        print(f"\n📊 Overall Metrics:")
        print(f"  Total Time: {total_time:.2f}s")
        print(f"  Total Requests: {total_requests}")
        print(f"  Successful: {self.metrics['successful_requests']} ({success_rate:.1f}%)")
        print(f"  Failed: {self.metrics['failed_requests']}")
        if total_time > 0:
            print(f"  Requests/Second: {total_requests / total_time:.2f}")
        
        # Session creation metrics
        if self.metrics["session_creation_times"]:
            print(f"\n🔑 Session Creation:")
            print(f"  Mean: {statistics.mean(self.metrics['session_creation_times']):.3f}s")
            print(f"  Median: {statistics.median(self.metrics['session_creation_times']):.3f}s")
            if len(self.metrics["session_creation_times"]) > 1:
                print(f"  Std Dev: {statistics.stdev(self.metrics['session_creation_times']):.3f}s")
        
        # Tool call metrics
        if self.metrics["tool_call_times"]:
            print(f"\n🔧 Tool Calls:")
            print(f"  Mean: {statistics.mean(self.metrics['tool_call_times']):.3f}s")
            print(f"  Median: {statistics.median(self.metrics['tool_call_times']):.3f}s")
            if len(self.metrics["tool_call_times"]) > 1:
                print(f"  Std Dev: {statistics.stdev(self.metrics['tool_call_times']):.3f}s")
                print(f"  Min: {min(self.metrics['tool_call_times']):.3f}s")
                print(f"  Max: {max(self.metrics['tool_call_times']):.3f}s")
                
                # Calculate percentiles
                sorted_times = sorted(self.metrics["tool_call_times"])
                p50_idx = len(sorted_times) // 2
                p95_idx = min(int(len(sorted_times) * 0.95), len(sorted_times) - 1)
                p99_idx = min(int(len(sorted_times) * 0.99), len(sorted_times) - 1)
                
                p50 = sorted_times[p50_idx]
                p95 = sorted_times[p95_idx]
                p99 = sorted_times[p99_idx]
                
                print(f"\n📈 Latency Percentiles:")
                print(f"  p50: {p50:.3f}s")
                print(f"  p95: {p95:.3f}s")
                print(f"  p99: {p99:.3f}s")
        
        # Throughput
        print(f"\n⚡ Throughput:")
        print(f"  Concurrent Clients: {num_clients}")
        if total_time > 0:
            print(f"  Requests/Second/Client: {(requests_per_client + 1) / total_time:.2f}")
            print(f"  Total Throughput: {total_requests / total_time:.2f} req/s")
        
        print(f"\n{'='*60}\n")


async def run_standard_tests(base_url: str = "http://localhost:7860", api_key: str = None):
    """Run standard load test scenarios."""
    tester = MCPLoadTester(base_url, api_key)
    
    # Test 1: Light load (10 clients, 5 requests each)
    print("\n🟢 TEST 1: Light Load")
    await tester.run_load_test(10, 5, delay=0.1)
    
    # Test 2: Medium load (50 clients, 10 requests each)
    print("\n🟡 TEST 2: Medium Load")
    await tester.run_load_test(50, 10, delay=0.05)
    
    # Test 3: Heavy load (100 clients, 20 requests each)
    print("\n🔴 TEST 3: Heavy Load")
    await tester.run_load_test(100, 20, delay=0.01)


async def run_stress_test(base_url: str = "http://localhost:7860", api_key: str = None):
    """Run stress test to find breaking point."""
    tester = MCPLoadTester(base_url, api_key)
    
    print("\n🔥 STRESS TEST: Finding Breaking Point")
    print("=" * 60)
    
    client_counts = [10, 25, 50, 100, 200, 500]
    results = []
    
    for clients in client_counts:
        print(f"\nTesting with {clients} concurrent clients...")
        metrics = await tester.run_load_test(clients, 10, delay=0.01)
        
        success_rate = (metrics["successful_requests"] / metrics["total_requests"] * 100) if metrics["total_requests"] > 0 else 0
        results.append((clients, success_rate))
        
        # Stop if success rate drops below 95%
        if success_rate < 95:
            print(f"\n⚠️ Performance degradation detected at {clients} clients")
            print(f"Success rate dropped to {success_rate:.1f}%")
            break
    
    print("\n📊 Stress Test Summary:")
    print("Clients | Success Rate")
    print("--------|-------------")
    for clients, rate in results:
        print(f"{clients:7d} | {rate:6.1f}%")


def main():
    parser = argparse.ArgumentParser(description='Load test W&B MCP Server')
    parser.add_argument('--url', default='http://localhost:7860', help='Server URL')
    parser.add_argument('--api-key', help='W&B API key (optional, uses test key if not provided)')
    parser.add_argument('--mode', choices=['standard', 'stress', 'custom'], default='standard',
                        help='Test mode: standard, stress, or custom')
    parser.add_argument('--clients', type=int, default=10, help='Number of concurrent clients (for custom mode)')
    parser.add_argument('--requests', type=int, default=10, help='Requests per client (for custom mode)')
    parser.add_argument('--delay', type=float, default=0.1, help='Delay between requests in seconds (for custom mode)')
    
    args = parser.parse_args()
    
    print("W&B MCP Server Load Tester")
    print(f"Server: {args.url}")
    print(f"Mode: {args.mode}")
    
    if args.mode == 'standard':
        asyncio.run(run_standard_tests(args.url, args.api_key))
    elif args.mode == 'stress':
        asyncio.run(run_stress_test(args.url, args.api_key))
    else:  # custom
        tester = MCPLoadTester(args.url, args.api_key)
        asyncio.run(tester.run_load_test(args.clients, args.requests, args.delay))


if __name__ == "__main__":
    main()
