$ErrorActionPreference = 'Continue'

Write-Host "=========================================================" -ForegroundColor Magenta
Write-Host "    AEOS 14-HOUR OVERNIGHT BATCH (VISION DATASET)        " -ForegroundColor Magenta
Write-Host "=========================================================" -ForegroundColor Magenta
Write-Host ""

Write-Host "[1/3] Migrating existing tabular results to results/tabular2/..." -ForegroundColor Yellow
python migrate_results.py
Write-Host "Migration complete.`n" -ForegroundColor Green

Write-Host "[2/3] Starting Vision Single Model Baselines..." -ForegroundColor Yellow
& .\run_exp3_local14b_vision.ps1

Write-Host "`n[3/3] Starting Vision Dual Agent Architectures..." -ForegroundColor Yellow
& .\run_exp3_dual_vision.ps1

Write-Host "`n=========================================================" -ForegroundColor Magenta
Write-Host "              ALL VISION EXPERIMENTS COMPLETE!             " -ForegroundColor Magenta
Write-Host "=========================================================" -ForegroundColor Magenta
