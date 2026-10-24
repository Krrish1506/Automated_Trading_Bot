# ─────────────────────────────────────────────────────────────
#  NSE LSTM Trading System — Environment Setup Script
#  Run once:  .\scripts\setup_env.ps1
# ─────────────────────────────────────────────────────────────

param(
    [switch]$DownloadData,   # Also download NSE data
    [switch]$RunTests        # Run tests after setup
)

$Root = Split-Path $PSScriptRoot -Parent

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "   NSE LSTM Trading System — Environment Setup" -ForegroundColor Cyan
Write-Host "═══════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

# 1. Verify Python
$PY = & python --version 2>&1
Write-Host "[1/5] Python: $PY" -ForegroundColor Green

# 2. Create venv if needed
$VenvPath = Join-Path $Root "venv"
if (-Not (Test-Path $VenvPath)) {
    Write-Host "[2/5] Creating virtual environment..." -ForegroundColor Yellow
    python -m venv $VenvPath
    Write-Host "      Created: $VenvPath" -ForegroundColor Green
} else {
    Write-Host "[2/5] Virtual environment already exists." -ForegroundColor Green
}

# 3. Upgrade pip
$Pip = Join-Path $VenvPath "Scripts\python.exe"
Write-Host "[3/5] Upgrading pip..." -ForegroundColor Yellow
& $Pip -m pip install --upgrade pip --quiet
Write-Host "      pip upgraded." -ForegroundColor Green

# 4. Install requirements
$ReqFile = Join-Path $Root "requirements.txt"
Write-Host "[4/5] Installing requirements..." -ForegroundColor Yellow
& $Pip -m pip install -r $ReqFile --quiet
if ($LASTEXITCODE -ne 0) {
    Write-Host "      [WARN] Some packages may have failed. Check output above." -ForegroundColor Yellow
} else {
    Write-Host "      All packages installed." -ForegroundColor Green
}

# 5. Create required directories
Write-Host "[5/5] Creating project directories..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path (Join-Path $Root "data")   | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $Root "models") | Out-Null
Write-Host "      data/ and models/ created." -ForegroundColor Green

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "   Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "   Next steps:" -ForegroundColor White
Write-Host "   1. Download NSE data:" -ForegroundColor White
Write-Host "      .\venv\Scripts\python.exe scripts\download_nifty.py" -ForegroundColor Yellow
Write-Host "   2. Run dataset validation tests:" -ForegroundColor White
Write-Host "      .\venv\Scripts\pytest.exe tests\test_nse_dataset.py -v" -ForegroundColor Yellow
Write-Host "   3. Train the model:" -ForegroundColor White
Write-Host "      .\venv\Scripts\python.exe -m forecast.train_lstm_v2" -ForegroundColor Yellow
Write-Host "   4. Run full test suite:" -ForegroundColor White
Write-Host "      .\venv\Scripts\pytest.exe tests\ -v" -ForegroundColor Yellow
Write-Host "═══════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

# Optional: download data
if ($DownloadData) {
    Write-Host "Downloading NSE data..." -ForegroundColor Cyan
    & $Pip scripts\download_nifty.py
}

# Optional: run tests
if ($RunTests) {
    Write-Host "Running NSE validation tests..." -ForegroundColor Cyan
    $Pytest = Join-Path $VenvPath "Scripts\pytest.exe"
    & $Pytest (Join-Path $Root "tests\test_nse_dataset.py") -v --tb=short
}
