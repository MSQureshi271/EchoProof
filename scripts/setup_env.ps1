param(
    [string]$Venv = ".venv",
    [switch]$SkipInstall
)
$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root
$Python = Get-Command python -ErrorAction SilentlyContinue
if (-not $Python) { throw "Python 3.11 or 3.13 is required but 'python' was not found on PATH." }
$Version = & python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
if ($Version -notin @("3.11", "3.13")) { throw "Expected Python 3.11 or 3.13; found $Version." }
$EnvPath = Join-Path $Root $Venv
if (-not (Test-Path (Join-Path $EnvPath "Scripts\python.exe"))) {
    & python -m venv $EnvPath
    if ($LASTEXITCODE -ne 0) { throw "Could not create venv at $EnvPath (exit $LASTEXITCODE)." }
}
$EnvPython = Join-Path $EnvPath "Scripts\python.exe"
if (-not $SkipInstall) {
    & $EnvPython -m pip install -r (Join-Path $Root "requirements\app.lock.txt")
    if ($LASTEXITCODE -ne 0) { throw "Application dependency install failed (exit $LASTEXITCODE)." }
    & $EnvPython -m pip install --no-deps --no-build-isolation -e $Root
    if ($LASTEXITCODE -ne 0) { throw "Editable EchoProof install failed (exit $LASTEXITCODE)." }
}
& $EnvPython (Join-Path $Root "scripts\check_app_env.py")
if ($LASTEXITCODE -ne 0) { throw "Import or pinned-version check failed (exit $LASTEXITCODE)." }
Write-Host "Environment ready: $EnvPath. Activate with .\.venv\Scripts\Activate.ps1"
