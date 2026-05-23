# AEOS Experiment Automation Script (Powershell)
# Features a Safety/Resume Mode: It will automatically skip any experiment that already has a JSON file in the results folder.

$ErrorActionPreference = "Stop"
$ResultsDir = "results"

Write-Host "=========================================================" -ForegroundColor Cyan
Write-Host " STARTING AEOS EXPERIMENT AUTOMATION (WITH RESUME MODE)" -ForegroundColor Cyan
Write-Host "=========================================================" -ForegroundColor Cyan
Write-Host ""

function Run-Experiment {
    param (
        [string]$StepName,
        [string]$Command,
        [string]$FilePattern
    )
    
    $ExistingFiles = Get-ChildItem -Path $ResultsDir -Filter $FilePattern -ErrorAction SilentlyContinue
    
    if ($ExistingFiles.Count -gt 0) {
        Write-Host "--> SKIP: $StepName (Found existing results: $($ExistingFiles[0].Name))" -ForegroundColor DarkGray
    } else {
        Write-Host "--> RUNNING: $StepName..." -ForegroundColor Yellow
        Invoke-Expression $Command
        Write-Host "--> $StepName Complete!`n" -ForegroundColor Green
    }
}

# =====================================================================
# PHASE 1: Single Agent Ground Truth (Remaining)
# =====================================================================
Run-Experiment -StepName "Step 1.3: Small Reviewer Baseline (phi3:mini)" -Command "python runner.py --backend ollama --model phi3:mini --dataset tabular2" -FilePattern "exp1_phi3-mini_tabular2_*.json"

# =====================================================================
# PHASE 2: Monolithic Baselines
# =====================================================================
Run-Experiment -StepName "Step 2.1: Deep Thinking Baseline (deepseek-r1:8b)" -Command "python runner.py --backend ollama --model deepseek-r1:8b --dataset tabular2" -FilePattern "exp1_deepseek-r1-8b_tabular2_*.json"

Run-Experiment -StepName "Step 2.2: Big Coder Baseline 1 (qwen2.5-coder:14b)" -Command "python runner.py --backend ollama --model qwen2.5-coder:14b --dataset tabular2" -FilePattern "exp1_qwen2.5-coder-14b_tabular2_*.json"

Run-Experiment -StepName "Step 2.3: Big Coder Baseline 2 (deepseek-coder-v2:16b)" -Command "python runner.py --backend ollama --model deepseek-coder-v2:16b --dataset tabular2" -FilePattern "exp1_deepseek-coder-v2-16b_tabular2_*.json"

# =====================================================================
# PHASE 3: Asymmetric Dual-Agent (Config B)
# =====================================================================
Run-Experiment -StepName "Step 3.1: Ultra-Small Diverse (phi3:mini + qwen2.5-coder:3b)" -Command "python runner_critic.py --reviewer-backend ollama --reviewer-model phi3:mini --coder-backend ollama --coder-model qwen2.5-coder:3b --dataset tabular2" -FilePattern "exp2_phi3-mini_qwen2.5-coder-3b_tabular2_*.json"

Run-Experiment -StepName "Step 3.2: Standard Diverse (phi3:mini + qwen2.5-coder:7b)" -Command "python runner_critic.py --reviewer-backend ollama --reviewer-model phi3:mini --coder-backend ollama --coder-model qwen2.5-coder:7b --dataset tabular2" -FilePattern "exp2_phi3-mini_qwen2.5-coder-7b_tabular2_*.json"

Run-Experiment -StepName "Step 3.3: Mid-Size Diverse (qwen2.5-coder:7b + llama3.1:8b)" -Command "python runner_critic.py --reviewer-backend ollama --reviewer-model qwen2.5-coder:7b --coder-backend ollama --coder-model llama3.1:8b --dataset tabular2" -FilePattern "exp2_qwen2.5-coder-7b_llama3.1-8b_tabular2_*.json"

# =====================================================================
# PHASE 4: Symmetric Dual-Agent (Config C)
# =====================================================================
Run-Experiment -StepName "Step 4.1: Homogeneous Ensemble (qwen2.5-coder:7b x 2)" -Command "python runner_critic.py --reviewer-backend ollama --reviewer-model qwen2.5-coder:7b --coder-backend ollama --coder-model qwen2.5-coder:7b --dataset tabular2" -FilePattern "exp2_qwen2.5-coder-7b_qwen2.5-coder-7b_tabular2_*.json"


Write-Host "=========================================================" -ForegroundColor Cyan
Write-Host " ALL EXPERIMENTS COMPLETE!" -ForegroundColor Cyan
Write-Host "=========================================================" -ForegroundColor Cyan
