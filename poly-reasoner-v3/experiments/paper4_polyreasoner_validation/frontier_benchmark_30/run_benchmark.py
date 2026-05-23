"""
Run this directly: python run_benchmark.py
Uses API keys from environment and launches the frontier benchmark.
"""
import os
import sys

required_keys = ["GROQ_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"]
missing_keys = [key for key in required_keys if not os.environ.get(key)]

print("=" * 60)
print("  Paper 3 Thread D — Frontier Benchmark")
print("  Models: Claude Sonnet 4.6, Claude Haiku 4.5,")
print("          GPT-4o, Llama-3.3-70B, Llama-4-Scout, Qwen3-32B")
print("  Mode: Resume — only runs missing/errored entries")
if missing_keys:
    print(f"  Warning: Missing env keys: {', '.join(missing_keys)}")
    print("           Missing providers will be marked as SKIP.")
print("=" * 60)
print()

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
sys.path.insert(0, script_dir)

import thread_d_frontier_benchmark

thread_d_frontier_benchmark.main()
