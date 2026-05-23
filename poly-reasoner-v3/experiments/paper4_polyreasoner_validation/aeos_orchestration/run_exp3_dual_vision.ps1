$ErrorActionPreference = 'Continue'

Write-Host ''
Write-Host '=========================================================' -ForegroundColor Cyan
Write-Host 'Starting AEOS EXP3 - Vision Dual Model Benchmarks' -ForegroundColor Cyan
Write-Host '=========================================================' -ForegroundColor Cyan

$dataset = 'vision'
$runs = 'run1', 'run2', 'run3'

$architectures = @(
    'qwen2.5-coder:3b|llama3.1:8b',
    'qwen2.5-coder:7b|qwen3.5:9b',
    'llama3.1:8b|qwen2.5-coder:3b',
    'qwen3.5:9b|qwen2.5-coder:7b',
    'deepseek-coder-v2:16b|qwen2.5-coder:14b'
)

foreach ($arch in $architectures) {
    $parts = $arch -split '\|'
    $coder = $parts[0]
    $reviewer = $parts[1]
    
    foreach ($r in $runs) {
        Write-Host "`n>>> RUNNING DUAL AGENT: Coder: $coder | Reviewer: $reviewer | Run: $r | Dataset: $dataset" -ForegroundColor Yellow
        python runner_critic.py --coder-model $coder --reviewer-model $reviewer --dataset $dataset --run-tag $r
    }
}

Write-Host ''
Write-Host '=========================================================' -ForegroundColor Green
Write-Host 'Vision Dual Model Benchmarks Complete!' -ForegroundColor Green
Write-Host '=========================================================' -ForegroundColor Green
