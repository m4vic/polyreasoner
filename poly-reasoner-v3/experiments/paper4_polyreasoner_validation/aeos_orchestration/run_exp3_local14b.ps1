$ErrorActionPreference = "Continue"

Write-Host "Starting EXP3 Local 14B Benchmark..." -ForegroundColor Cyan

$models = "qwen2.5-coder:14b", "deepseek-coder-v2:16b", "qwen3.5:9b", "ministral-3:14b"
$runs = "run1", "run2", "run3"

foreach ($m in $models) {
    foreach ($r in $runs) {
        Write-Host ">>> RUNNING: $m ($r)" -ForegroundColor Yellow
        python runner.py --backend ollama --model $m --dataset tabular2 --run-tag $r
    }
}

Write-Host "EXP3 Local 14B Benchmark Complete." -ForegroundColor Green
