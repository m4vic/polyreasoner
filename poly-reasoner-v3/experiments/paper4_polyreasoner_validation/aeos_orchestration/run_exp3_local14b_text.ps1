$ErrorActionPreference = "Continue"

Write-Host "=========================================================" -ForegroundColor Cyan
Write-Host "Starting EXP3 Local Benchmark (TEXT DATASET) - ALL MODELS" -ForegroundColor Cyan
Write-Host "=========================================================" -ForegroundColor Cyan

$models = @(
    "deepseek-coder-v2:16b",
    "llama3.1:8b",
    "ministral-3:14b",
    "phi3:mini",
    "qwen2.5-coder:14b",
    "qwen2.5-coder:3b",
    "qwen2.5-coder:7b",
    "qwen3.5:9b"
)
$runs = "run1", "run2", "run3"

foreach ($m in $models) {
    foreach ($r in $runs) {
        Write-Host ""
        Write-Host ">>> RUNNING SINGLE MODEL: $m | Run: $r | Dataset: text" -ForegroundColor Yellow
        python runner.py --backend ollama --model $m --dataset text --run-tag $r
    }
}

Write-Host ""
Write-Host "=========================================================" -ForegroundColor Green
Write-Host "EXP3 Local Benchmark (TEXT) Complete." -ForegroundColor Green
Write-Host "=========================================================" -ForegroundColor Green
