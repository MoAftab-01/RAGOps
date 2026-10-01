<#
.SYNOPSIS
    RAGOps developer helper for Windows PowerShell.

.DESCRIPTION
    Every target mirrors a Makefile target, so nothing here is required to use
    RAGOps -- this is a convenience wrapper for the primary target environment.

.EXAMPLE
    ./scripts/dev.ps1 setup        # create venv, install deps, start infra
    ./scripts/dev.ps1 migrate      # alembic upgrade head
    ./scripts/dev.ps1 test         # backend + frontend test suites
    ./scripts/dev.ps1 demo-data    # generate 10,000+ traces
#>
[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet('setup', 'infra', 'infra-down', 'migrate', 'migrate-down',
                 'reset-db', 'seed', 'backend', 'frontend', 'test',
                 'test-backend', 'test-frontend', 'lint', 'typecheck',
                 'demo-data', 'demo-rag', 'demo-agent', 'verify', 'clean',
                 'help')]
    [string]$Task = 'help'
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$Py = Join-Path $Root '.venv\Scripts\python.exe'

function Write-Step { param([string]$Message) Write-Host "==> $Message" -ForegroundColor Cyan }

function Assert-Venv {
    if (-not (Test-Path $Py)) {
        throw "Virtual environment not found at $Py. Run: ./scripts/dev.ps1 setup"
    }
}

switch ($Task) {
    'setup' {
        Write-Step 'Creating virtual environment'
        if (-not (Test-Path $Py)) { python -m venv (Join-Path $Root '.venv') }
        Write-Step 'Installing backend dependencies (this pulls torch, be patient)'
        & (Join-Path $Root '.venv\Scripts\pip.exe') install -r backend/requirements.txt -r backend/requirements-ml.txt
        Write-Step 'Installing frontend dependencies'
        Push-Location (Join-Path $Root 'frontend'); npm install; Pop-Location
        Write-Step 'Starting infrastructure'
        docker compose up -d postgres redis qdrant
        Write-Step 'Done. Next: ./scripts/dev.ps1 migrate'
    }

    'infra'        { docker compose up -d postgres redis qdrant }
    'infra-down'   { docker compose down }
    'migrate'      { Assert-Venv; Push-Location (Join-Path $Root 'backend'); & (Join-Path $Root '.venv\Scripts\alembic.exe') -c alembic.ini upgrade head; Pop-Location }
    'migrate-down' { Assert-Venv; Push-Location (Join-Path $Root 'backend'); & (Join-Path $Root '.venv\Scripts\alembic.exe') -c alembic.ini downgrade -1; Pop-Location }
    'reset-db'     { Assert-Venv; Push-Location (Join-Path $Root 'backend'); & (Join-Path $Root '.venv\Scripts\alembic.exe') -c alembic.ini downgrade base; & (Join-Path $Root '.venv\Scripts\alembic.exe') -c alembic.ini upgrade head; Pop-Location }
    'seed'         { Assert-Venv; Push-Location $Root; & $Py scripts/seed_models.py; Pop-Location }

    'backend'  { Assert-Venv; Push-Location (Join-Path $Root 'backend'); & $Py -m uvicorn app.main:app --reload --port 8000; Pop-Location }
    'frontend' { Push-Location (Join-Path $Root 'frontend'); npm run dev; Pop-Location }

    'test'          { & $PSScriptRoot\dev.ps1 test-backend; & $PSScriptRoot\dev.ps1 test-frontend }
    'test-backend'  { Assert-Venv; Push-Location (Join-Path $Root 'backend'); & $Py -m pytest -q; Pop-Location }
    'test-frontend' { Push-Location (Join-Path $Root 'frontend'); npm run test; Pop-Location }

    'lint'      { Assert-Venv; Push-Location (Join-Path $Root 'backend'); & $Py -m ruff check app tests; & $Py -m mypy app; Pop-Location }
    'typecheck' { Push-Location (Join-Path $Root 'frontend'); npm run typecheck; Pop-Location }

    'demo-data'  { Assert-Venv; Push-Location $Root; & $Py scripts/generate_demo_data.py; Pop-Location }
    'demo-rag'   { Assert-Venv; Push-Location (Join-Path $Root 'evaluation'); & $Py -m rag_demo.app --query 'How do I reset my password?'; Pop-Location }
    'demo-agent' { Assert-Venv; Push-Location (Join-Path $Root 'evaluation'); & $Py -m agent_demo.app; Pop-Location }

    'verify' { & $PSScriptRoot\dev.ps1 migrate; & $PSScriptRoot\dev.ps1 test }

    'clean' {
        Get-ChildItem -Recurse -Directory -Filter '__pycache__' | Remove-Item -Recurse -Force
        Get-ChildItem -Recurse -Directory -Filter '.pytest_cache' | Remove-Item -Recurse -Force
        $dist = Join-Path $Root 'frontend\dist'
        if (Test-Path $dist) { Remove-Item $dist -Recurse -Force }
    }

    'help' {
        Get-Content $PSCommandPath | Select-Object -First 20
        Write-Host ''
        Write-Host 'Targets: setup infra infra-down migrate migrate-down reset-db seed' -ForegroundColor Gray
        Write-Host '         backend frontend test test-backend test-frontend' -ForegroundColor Gray
        Write-Host '         lint typecheck demo-data demo-rag demo-agent verify clean' -ForegroundColor Gray
    }
}
