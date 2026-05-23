@echo off
echo ============================================================
echo  Paper 3 Thread D - Frontier Benchmark
echo ============================================================

cd /d f:\AI-IN-THE-LOOP\aitl-paper\experiments\aeos\aeos_behave\paper3_thread_d

echo.
echo [STATUS] Starting benchmark - will retry errored models only
echo [STATUS] Results checkpoint: frontier_benchmark_results_v3.json
echo [STATUS] API keys must be preset in your environment:
echo          GROQ_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY
echo.

python thread_d_frontier_benchmark.py

echo.
echo [DONE] Check frontier_benchmark_results_v3.json for results
pause
