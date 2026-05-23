$ErrorActionPreference = "Continue"

if (-not $env:OPENAI_API_KEY) {
    $envFile = Join-Path $PSScriptRoot ".env"
    if (Test-Path $envFile) {
        Get-Content $envFile | ForEach-Object {
            if ($_ -match "^\s*OPENAI_API_KEY\s*=\s*(.+)$") {
                $env:OPENAI_API_KEY = $matches[1].Trim()
                Write-Host "[.env] Loaded OPENAI_API_KEY" -ForegroundColor Green
            }
        }
    }
}

if (-not $env:OPENAI_API_KEY) {
    Write-Host "ERROR: OPENAI_API_KEY not set." -ForegroundColor Red
    exit 1
}

Write-Host "Starting EXP3 GPT API Benchmark (N=1)..." -ForegroundColor Magenta

$models = "gpt-4o", "gpt-4.1"

foreach ($m in $models) {
    Write-Host ">>> RUNNING: [openai] $m (run1)" -ForegroundColor Yellow
    python runner.py --backend openai --model $m --dataset tabular2 --run-tag run1
}

Write-Host "EXP3 GPT BATCH COMPLETE." -ForegroundColor Magenta
