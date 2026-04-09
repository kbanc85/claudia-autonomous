#Requires -Version 5.1
<#
.SYNOPSIS
    Claudia Autonomous setup script for Windows.

.DESCRIPTION
    PowerShell equivalent of setup-claudia.sh. Assumes the repo has been
    cloned and you're running this from inside it (or install.ps1 called it).

    What this script does:
      1. Installs uv (via the official installer) if not present
      2. Provisions Python 3.11 via uv if needed
      3. Creates a virtual environment at .\venv
      4. Installs the claudia-autonomous package in editable mode with all extras
      5. Creates a claudia.cmd shim at %USERPROFILE%\.local\bin\claudia.cmd
         that forwards to the venv's claudia.exe
      6. Adds %USERPROFILE%\.local\bin to the user PATH (persistent)
      7. Copies the .env template if .env doesn't exist
      8. Seeds bundled skills into %CLAUDIA_HOME%\skills (or %USERPROFILE%\.claudia\skills)

    This script is idempotent: re-running it updates the venv and package
    but does not delete any user data.

.PARAMETER NonInteractive
    Skip all prompts. Used when called by install.ps1 during curl | iex flow.

.NOTES
    Author: Kamil Banc (https://github.com/kbanc85)
    License: MIT
#>

[CmdletBinding()]
param(
    [switch]$NonInteractive
)

$ErrorActionPreference = 'Stop'

# ============================================================================
# Locate script directory (works whether invoked via -File or dot-sourced)
# ============================================================================

$ScriptDir = if ($PSScriptRoot) {
    $PSScriptRoot
} elseif ($MyInvocation.MyCommand.Path) {
    Split-Path -Parent $MyInvocation.MyCommand.Path
} else {
    (Get-Location).Path
}
Set-Location $ScriptDir

# ============================================================================
# Helpers
# ============================================================================

function Write-Info { param([string]$Message) Write-Host "==> " -ForegroundColor Cyan -NoNewline; Write-Host $Message }
function Write-Warn { param([string]$Message) Write-Host "!   " -ForegroundColor Yellow -NoNewline; Write-Host $Message }
function Write-Ok   { param([string]$Message) Write-Host "OK  " -ForegroundColor Green -NoNewline; Write-Host $Message }
function Write-Err  { param([string]$Message) Write-Host "!!  " -ForegroundColor Red -NoNewline; Write-Host $Message }
function Fail       { param([string]$Message) Write-Err $Message; exit 1 }

function Test-Command {
    param([string]$Name)
    $null -ne (Get-Command $Name -ErrorAction SilentlyContinue)
}

# ============================================================================
# Header
# ============================================================================

Write-Host ""
Write-Host "Claudia Setup (Windows)" -ForegroundColor Cyan
Write-Host "Working in: $ScriptDir" -ForegroundColor DarkGray
Write-Host ""

$PythonVersion = '3.11'

# ============================================================================
# Install / locate uv
# ============================================================================

Write-Info "Checking for uv..."

$uvCmd = $null
$uvCandidates = @(
    'uv',
    (Join-Path $env:USERPROFILE '.cargo\bin\uv.exe'),
    (Join-Path $env:USERPROFILE '.local\bin\uv.exe')
)

foreach ($candidate in $uvCandidates) {
    if ($candidate -eq 'uv') {
        if (Test-Command 'uv') { $uvCmd = 'uv'; break }
    } elseif (Test-Path $candidate) {
        $uvCmd = $candidate
        break
    }
}

if ($uvCmd) {
    try {
        $uvVersion = (& $uvCmd --version 2>$null).Trim()
        Write-Ok "uv found ($uvVersion)"
    } catch {
        Write-Ok "uv found (version check failed, continuing)"
    }
} else {
    Write-Info "Installing uv via the official installer..."
    try {
        # Official uv Windows installer. Downloads to %USERPROFILE%\.cargo\bin\uv.exe
        # or %USERPROFILE%\.local\bin\uv.exe depending on the installer version.
        $uvInstallScript = (Invoke-WebRequest -UseBasicParsing -Uri 'https://astral.sh/uv/install.ps1').Content
        Invoke-Expression $uvInstallScript
    } catch {
        Fail "Failed to download or run the uv installer: $_"
    }

    # Re-scan for uv in expected locations
    foreach ($candidate in $uvCandidates) {
        if ($candidate -eq 'uv') {
            # Refresh PATH and retry
            $env:Path = [System.Environment]::GetEnvironmentVariable('Path','Machine') + ";" + [System.Environment]::GetEnvironmentVariable('Path','User')
            if (Test-Command 'uv') { $uvCmd = 'uv'; break }
        } elseif (Test-Path $candidate) {
            $uvCmd = $candidate
            break
        }
    }

    if (-not $uvCmd) {
        Fail "uv installed but not found at any expected location. Please restart PowerShell and re-run setup-claudia.ps1."
    }

    try {
        $uvVersion = (& $uvCmd --version 2>$null).Trim()
        Write-Ok "uv installed ($uvVersion)"
    } catch {
        Write-Ok "uv installed"
    }
}

# ============================================================================
# Ensure Python 3.11
# ============================================================================

Write-Info "Checking Python $PythonVersion..."

$pythonPath = $null
try {
    $pythonPath = (& $uvCmd python find $PythonVersion 2>$null | Out-String).Trim()
} catch { }

if (-not $pythonPath) {
    Write-Info "Python $PythonVersion not found via uv. Installing..."
    & $uvCmd python install $PythonVersion
    if ($LASTEXITCODE -ne 0) {
        Fail "uv python install $PythonVersion failed"
    }
    $pythonPath = (& $uvCmd python find $PythonVersion | Out-String).Trim()
}

if ($pythonPath) {
    try {
        $pythonVerString = (& $pythonPath --version 2>&1 | Out-String).Trim()
        Write-Ok "$pythonVerString at $pythonPath"
    } catch {
        Write-Ok "Python ready at $pythonPath"
    }
} else {
    Fail "Unable to provision Python $PythonVersion"
}

# ============================================================================
# Virtual environment
# ============================================================================

Write-Info "Setting up virtual environment..."

$venvDir = Join-Path $ScriptDir 'venv'

if (Test-Path $venvDir) {
    Write-Info "Removing old venv..."
    Remove-Item -Recurse -Force $venvDir
}

& $uvCmd venv venv --python $PythonVersion
if ($LASTEXITCODE -ne 0) {
    Fail "uv venv creation failed"
}
Write-Ok "venv created (Python $PythonVersion)"

# Tell uv where to install packages
$env:VIRTUAL_ENV = $venvDir
$env:UV_PROJECT_ENVIRONMENT = $venvDir

# ============================================================================
# Install dependencies
# ============================================================================

Write-Info "Installing dependencies (this takes a minute)..."

$installSucceeded = $false
if (Test-Path 'uv.lock') {
    Write-Info "Trying lockfile-verified install (uv sync)..."
    try {
        & $uvCmd sync --all-extras --locked
        if ($LASTEXITCODE -eq 0) {
            Write-Ok "Dependencies installed (lockfile verified)"
            $installSucceeded = $true
        }
    } catch {
        Write-Warn "uv sync failed: $_"
    }
}

if (-not $installSucceeded) {
    if (Test-Path 'uv.lock') {
        Write-Warn "Lockfile install failed, falling back to uv pip install..."
    }
    try {
        & $uvCmd pip install -e ".[all]"
        if ($LASTEXITCODE -eq 0) {
            Write-Ok "Dependencies installed (with [all] extras)"
            $installSucceeded = $true
        }
    } catch { }

    if (-not $installSucceeded) {
        Write-Warn "Full-extras install failed, trying core-only install..."
        & $uvCmd pip install -e "."
        if ($LASTEXITCODE -ne 0) {
            Fail "Package install failed. See errors above."
        }
        Write-Ok "Dependencies installed (core only)"
    }
}

# ============================================================================
# .env file
# ============================================================================

if (-not (Test-Path '.env')) {
    if (Test-Path '.env.example') {
        Copy-Item '.env.example' '.env'
        Write-Ok "Created .env from template"
    }
} else {
    Write-Ok ".env exists"
}

# ============================================================================
# Create claudia.cmd shim in %USERPROFILE%\.local\bin
# ============================================================================

Write-Info "Setting up claudia command..."

$binDir = Join-Path $env:USERPROFILE '.local\bin'
if (-not (Test-Path $binDir)) {
    New-Item -Path $binDir -ItemType Directory -Force | Out-Null
}

$claudiaExe = Join-Path $venvDir 'Scripts\claudia.exe'
$shimPath = Join-Path $binDir 'claudia.cmd'

if (-not (Test-Path $claudiaExe)) {
    Fail "claudia.exe not found at $claudiaExe. The package install may have failed."
}

# Write a minimal .cmd shim that forwards all args to the venv executable.
# Using ASCII encoding and CRLF line endings, which is what Windows cmd expects.
$shimContent = "@echo off`r`n`"$claudiaExe`" %*`r`n"
Set-Content -Path $shimPath -Value $shimContent -Encoding ASCII -NoNewline

Write-Ok "Created claudia.cmd at $shimPath"

# ============================================================================
# Update user PATH (persistent, not just current session)
# ============================================================================

$userPath = [System.Environment]::GetEnvironmentVariable('Path', 'User')
if (-not $userPath) { $userPath = '' }

$pathEntries = $userPath -split ';' | Where-Object { $_ -and ($_.Trim() -ne '') }
$alreadyOnPath = $pathEntries | Where-Object { $_.TrimEnd('\') -ieq $binDir.TrimEnd('\') }

if (-not $alreadyOnPath) {
    $newPath = if ($userPath) { "$userPath;$binDir" } else { $binDir }
    [System.Environment]::SetEnvironmentVariable('Path', $newPath, 'User')
    Write-Ok "Added $binDir to user PATH (persistent)"
    Write-Warn "Restart PowerShell for the PATH change to take effect in existing sessions."
} else {
    Write-Ok "$binDir already on user PATH"
}

# Also update the current session so immediate testing is possible
if ($env:Path -notmatch [regex]::Escape($binDir)) {
    $env:Path = "$env:Path;$binDir"
}

# ============================================================================
# Seed bundled skills into %CLAUDIA_HOME%\skills
# ============================================================================

$claudiaHome = if ($env:CLAUDIA_HOME) { $env:CLAUDIA_HOME } else { Join-Path $env:USERPROFILE '.claudia' }
$claudiaSkillsDir = Join-Path $claudiaHome 'skills'

if (-not (Test-Path $claudiaSkillsDir)) {
    New-Item -Path $claudiaSkillsDir -ItemType Directory -Force | Out-Null
}

Write-Host ""
Write-Info "Syncing bundled skills to $claudiaSkillsDir ..."

$skillsSyncScript = Join-Path $ScriptDir 'tools\skills_sync.py'
$bundledSkillsDir = Join-Path $ScriptDir 'skills'
$venvPython       = Join-Path $venvDir 'Scripts\python.exe'

$synced = $false
if ((Test-Path $skillsSyncScript) -and (Test-Path $venvPython)) {
    try {
        & $venvPython $skillsSyncScript
        if ($LASTEXITCODE -eq 0) {
            Write-Ok "Skills synced via tools/skills_sync.py"
            $synced = $true
        }
    } catch {
        Write-Warn "Skills sync script failed: $_"
    }
}

if (-not $synced) {
    # Fallback: plain copy
    if (Test-Path $bundledSkillsDir) {
        try {
            Copy-Item -Recurse -Force (Join-Path $bundledSkillsDir '*') $claudiaSkillsDir
            Write-Ok "Skills copied (fallback)"
        } catch {
            Write-Warn "Skills copy failed: $_"
        }
    }
}

# ============================================================================
# Done
# ============================================================================

Write-Host ""
Write-Host "OK  Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Next steps:" -ForegroundColor White
Write-Host ""
Write-Host "  1. Restart PowerShell (so $binDir is on PATH)" -ForegroundColor White
Write-Host ""
Write-Host "  2. Run the Claudia setup wizard to configure API keys:" -ForegroundColor White
Write-Host "     claudia setup"
Write-Host ""
Write-Host "  3. Activate the hybrid memory provider:" -ForegroundColor White
Write-Host "     claudia memory setup"
Write-Host ""
Write-Host "  4. Verify everything is wired up:" -ForegroundColor White
Write-Host "     claudia doctor"
Write-Host ""
Write-Host "  5. Start chatting:" -ForegroundColor White
Write-Host "     claudia"
Write-Host ""

exit 0
