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

Write-Host "Starting EXP3 Dual-Agent Benchmark..." -ForegroundColor Cyan

$combos = @(
    @("openai", "gpt-4o", "ollama", "qwen2.5-coder:14b", "gpt4o-rev_qwen14b-coder"),
    @("ollama", "qwen2.5-coder:14b", "openai", "gpt-4o", "qwen14b-rev_gpt4o-coder"),
    @("openai", "gpt-4.1", "ollama", "deepseek-coder-v2:16b", "gpt41-rev_dscoder16b-coder"),
    @("ollama", "qwen3.5:9b", "ollama", "qwen2.5-coder:14b", "qwen35think-rev_qwen14b-coder")
)

foreach ($c in $combos) {
    $rev_backend = $c[0]
    $rev_model = $c[1]
    $cod_backend = $c[2]
    $cod_model = $c[3]
    $label = $c[4]

    Write-Host ">>> RUNNING: [$rev_backend] $rev_model + [$cod_backend] $cod_model" -ForegroundColor Yellow
    python runner_critic.py --reviewer-backend $rev_backend --reviewer-model $rev_model --coder-backend $cod_backend --coder-model $cod_model --dataset tabular2 --run-tag run1
}

Write-Host "EXP3 DUAL-AGENT BATCH COMPLETE." -ForegroundColor Cyan
