# PowerShell: Activate venv for Legal RAG Backend
# Usage: .\scripts\activate_env.ps1

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$VenvPath = Join-Path $ProjectRoot ".venv\Scripts\Activate.ps1"

if (Test-Path $VenvPath) {
    & $VenvPath
    Write-Host "✓ Virtual environment activated" -ForegroundColor Green
    Write-Host "  Python: $((Get-Command python).Source)" -ForegroundColor Cyan
} else {
    Write-Host "✗ Virtual environment not found at: $VenvPath" -ForegroundColor Red
    Write-Host "  Run: python -m venv .venv" -ForegroundColor Yellow
}
