$ErrorActionPreference = 'Continue'

Write-Host ''
Write-Host '=========================================================' -ForegroundColor Cyan
Write-Host ' AEOS Architecture D - Tri-Agent Overnight' -ForegroundColor Cyan
Write-Host ' Multi-Perspective Reasoning (3 Models)' -ForegroundColor Cyan
Write-Host '=========================================================' -ForegroundColor Cyan

$datasets = 'tabular2', 'text', 'vision'
$runs = 'run1', 'run2', 'run3'

# =====================================================================
# CONFIG 1: Asymmetric Small vs Mid (Diverse Coders)
# Judge: qwen3.5:9b | Coder A: qwen2.5-coder:3b | Coder B: qwen2.5-coder:7b
# =====================================================================
Write-Host ''
Write-Host '--- CONFIG 1: Judge=qwen3.5:9b | A=3b | B=7b (diverse coders) ---' -ForegroundColor Yellow

foreach ($ds in $datasets) {
    foreach ($r in $runs) {
        Write-Host ">>> TRI [$ds] $r : qwen3.5:9b | 3b vs 7b" -ForegroundColor Yellow
        python runner_tri_agent.py --judge-model qwen3.5:9b --coder-a-model qwen2.5-coder:3b --coder-b-model qwen2.5-coder:7b --dataset $ds --run-tag $r
    }
}

# =====================================================================
# CONFIG 2: Heavyweight Diverse Architectures
# Judge: llama3.1:8b | Coder A: qwen2.5-coder:7b | Coder B: deepseek-coder-v2:16b
# =====================================================================
Write-Host ''
Write-Host '--- CONFIG 2: Judge=llama3.1:8b | A=7b | B=16b (diverse coders) ---' -ForegroundColor Yellow

foreach ($ds in $datasets) {
    foreach ($r in $runs) {
        Write-Host ">>> TRI [$ds] $r : llama3.1:8b | 7b vs 16b" -ForegroundColor Yellow
        python runner_tri_agent.py --judge-model llama3.1:8b --coder-a-model qwen2.5-coder:7b --coder-b-model deepseek-coder-v2:16b --dataset $ds --run-tag $r
    }
}

# =====================================================================
# CONFIG 3: Small Judge + 2x Same Mid Coder
# Judge: qwen2.5-coder:3b | Coder A: qwen2.5-coder:7b | Coder B: qwen2.5-coder:7b
# Tests: Does competitive sampling from SAME weights help?
# =====================================================================
Write-Host ''
Write-Host '--- CONFIG 3: Judge=3b | A=7b | B=7b (same coders) ---' -ForegroundColor Magenta

foreach ($ds in $datasets) {
    foreach ($r in $runs) {
        Write-Host ">>> TRI [$ds] $r : qwen2.5-coder:3b | 7b vs 7b" -ForegroundColor Magenta
        python runner_tri_agent.py --judge-model qwen2.5-coder:3b --coder-a-model qwen2.5-coder:7b --coder-b-model qwen2.5-coder:7b --dataset $ds --run-tag $r
    }
}

# =====================================================================
# CONFIG 4: Tiny Judge + 2x Same Big Coder
# Judge: phi3:mini | Coder A: qwen2.5-coder:14b | Coder B: qwen2.5-coder:14b
# Tests: Can a tiny 3.8B judge steer two 14B coders effectively?
# =====================================================================
Write-Host ''
Write-Host '--- CONFIG 4: Judge=phi3:mini | A=14b | B=14b (same coders) ---' -ForegroundColor Magenta

foreach ($ds in $datasets) {
    foreach ($r in $runs) {
        Write-Host ">>> TRI [$ds] $r : phi3:mini | 14b vs 14b" -ForegroundColor Magenta
        python runner_tri_agent.py --judge-model phi3:mini --coder-a-model qwen2.5-coder:14b --coder-b-model qwen2.5-coder:14b --dataset $ds --run-tag $r
    }
}

# =====================================================================
# CONFIG 5: Small Thinker Judge + 2x Same Mid Coder
# Judge: qwen3.5:4b | Coder A: qwen2.5-coder:7b | Coder B: qwen2.5-coder:7b
# Tests: Does a reasoning-capable small judge outperform a code-only small judge (Config 3)?
# =====================================================================
Write-Host ''
Write-Host '--- CONFIG 5: Judge=qwen3.5:4b | A=7b | B=7b (same coders, thinker judge) ---' -ForegroundColor Magenta

foreach ($ds in $datasets) {
    foreach ($r in $runs) {
        Write-Host ">>> TRI [$ds] $r : qwen3.5:4b | 7b vs 7b" -ForegroundColor Magenta
        python runner_tri_agent.py --judge-model qwen3.5:4b --coder-a-model qwen2.5-coder:7b --coder-b-model qwen2.5-coder:7b --dataset $ds --run-tag $r
    }
}

Write-Host ''
Write-Host '=========================================================' -ForegroundColor Green
Write-Host ' ALL TRI-AGENT EXPERIMENTS COMPLETE!' -ForegroundColor Green
Write-Host ' Total: 5 configs x 3 datasets x 3 runs = 45 runs' -ForegroundColor Green
Write-Host '=========================================================' -ForegroundColor Green
