$ErrorActionPreference = 'Continue'

Write-Host "=========================================================" -ForegroundColor Magenta
Write-Host "    AEOS OVERNIGHT BATCH (TEXT DATASET)                  " -ForegroundColor Magenta
Write-Host "=========================================================" -ForegroundColor Magenta
Write-Host ""

Write-Host "[1/3] Ensuring text results directory exists..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path "results\text" | Out-Null
Write-Host "Text results will be saved under results/text/.`n" -ForegroundColor Green

Write-Host "[2/3] Starting Text Single Model Baselines..." -ForegroundColor Yellow
& .\run_exp3_local14b_text.ps1

Write-Host "`n[3/3] Starting Text Dual Agent Architectures..." -ForegroundColor Yellow
& .\run_exp3_dual_text.ps1

Write-Host "`n=========================================================" -ForegroundColor Magenta
Write-Host "              ALL TEXT EXPERIMENTS COMPLETE!             " -ForegroundColor Magenta
Write-Host "=========================================================" -ForegroundColor Magenta
