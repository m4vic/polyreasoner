$ErrorActionPreference = "Stop"

Write-Host "=========================================================" -ForegroundColor Cyan
Write-Host " STARTING N=3 TABULAR EXPERIMENT MATRIX (WITH RESUME MODE)" -ForegroundColor Cyan
Write-Host "=========================================================" -ForegroundColor Cyan

# ─── Configuration ───
$DATASET = "tabular2"
$RUNS = 3

# Baseline Models
$BASELINE_MODELS = @(
    "phi3:mini",
    "qwen2.5-coder:3b",
    "qwen2.5-coder:7b",
    "llama3.1:8b",
    "deepseek-r1:8b",
    "qwen2.5-coder:14b",
    "deepseek-coder-v2:16b"
)

# Dual-Agent Configurations (Reviewer, Coder)
$DUAL_MODELS = @(
    @("phi3:mini", "qwen2.5-coder:3b"),
    @("phi3:mini", "qwen2.5-coder:7b"),
    @("qwen2.5-coder:7b", "llama3.1:8b"),
    @("qwen2.5-coder:7b", "qwen2.5-coder:7b")
)

# ─── Helper function to safely check for existing files ───
function Run-Experiment {
    param (
        [string]$Name,
        [string]$FilePattern,
        [string]$Command
    )
    
    $existing = Get-ChildItem -Path "results" -Filter $FilePattern -ErrorAction SilentlyContinue
    if ($existing.Count -gt 0) {
        $found = $existing[0].Name
        Write-Host "--> SKIP: $Name (Found existing results: $found)" -ForegroundColor DarkGray
    } else {
        Write-Host "--> RUNNING: $Name" -ForegroundColor Green
        Write-Host "    $Command" -ForegroundColor DarkGray
        Invoke-Expression $Command
        
        # Adding a 5 second cool down between runs
        Write-Host "    [Cooling down for 5 seconds...]" -ForegroundColor DarkGray
        Start-Sleep -Seconds 5
    }
}

for ($i = 1; $i -le $RUNS; $i++) {
    $RunTag = "run$i"
    Write-Host "`n=========================================================" -ForegroundColor Magenta
    Write-Host " EXECUTING BATCH: $RunTag " -ForegroundColor Magenta
    Write-Host "=========================================================" -ForegroundColor Magenta

    # 1. Monolithic Baselines
    foreach ($model in $BASELINE_MODELS) {
        $safe_model = $model -replace ":", "-" -replace "/", "_"
        $file_pattern = "exp1_$safe_model`_$DATASET`_$RunTag`_*.json"
        
        Run-Experiment `
            -Name "Baseline ($model) - $RunTag" `
            -FilePattern $file_pattern `
            -Command "python runner.py --model $model --dataset $DATASET --run-tag $RunTag"
    }

    # 2. Dual-Agent Architectures
    foreach ($pair in $DUAL_MODELS) {
        $rev = $pair[0]
        $cod = $pair[1]
        
        $safe_rev = $rev -replace ":", "-" -replace "/", "_"
        $safe_cod = $cod -replace ":", "-" -replace "/", "_"
        $file_pattern = "exp2_$safe_rev`_$safe_cod`_$DATASET`_$RunTag`_*.json"
        
        Run-Experiment `
            -Name "Dual-Agent ($rev + $cod) - $RunTag" `
            -FilePattern $file_pattern `
            -Command "python runner_critic.py --reviewer-model $rev --coder-model $cod --dataset $DATASET --run-tag $RunTag"
    }
}

Write-Host "=========================================================" -ForegroundColor Cyan
Write-Host " ALL N=3 EXPERIMENTS COMPLETE!" -ForegroundColor Cyan
Write-Host "=========================================================" -ForegroundColor Cyan
