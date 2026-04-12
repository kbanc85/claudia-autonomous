#Requires -Version 5.1
<#
    Claudia Autonomous - Windows one-line installer

    Bootstrap installer for Claudia Autonomous on Windows 10/11. Installs
    prerequisites (git, Python 3.11+, Ollama) via winget if missing, clones
    the repo, runs setup-claudia.ps1 for the venv + package install, and
    pulls the default Ollama models for the Claudia Hybrid Memory plugin.

    Recommended usage (one-line install):

        iwr -useb https://raw.githubusercontent.com/kbanc85/claudia-autonomous/main/install.ps1 | iex

    The 'iwr | iex' pattern runs the script content as an in-memory
    expression. Unlike loading a script from disk, it bypasses PowerShell's
    execution policy, so users do not need to change their execution policy
    to install Claudia. This matches how Scoop, uv, Starship, and other
    CLI tools ship their Windows installers.

    IMPORTANT: Because 'iex' cannot pass named parameters, customization
    is done via environment variables, set BEFORE the install command:

        $env:INSTALL_DIR = 'D:\dev\claudia'
        $env:BRANCH      = 'v1.0.0'
        iwr -useb https://raw.githubusercontent.com/kbanc85/claudia-autonomous/main/install.ps1 | iex

    Supported env vars:
        INSTALL_DIR    where to clone the repo (default: $env:USERPROFILE\claudia-autonomous)
        REPO_URL       git URL (default: https://github.com/kbanc85/claudia-autonomous.git)
        BRANCH         branch or tag (default: main)
        SKIP_OLLAMA    any value = skip Ollama install + model pull
        SKIP_SETUP     any value = skip running setup-claudia.ps1
        SKIP_MODELS    any value = skip model pull only (still installs Ollama)

    Author: Kamil Banc (https://github.com/kbanc85)
    License: MIT
#>

$ErrorActionPreference = 'Stop'

# ============================================================================
# Configuration (from environment variables, with sane defaults)
# ============================================================================

$InstallDir = if ($env:INSTALL_DIR) { $env:INSTALL_DIR } else { Join-Path $env:USERPROFILE 'claudia-autonomous' }
$RepoUrl    = if ($env:REPO_URL)    { $env:REPO_URL }    else { 'https://github.com/kbanc85/claudia-autonomous.git' }
$Branch     = if ($env:BRANCH)      { $env:BRANCH }      else { 'main' }
$SkipOllama = [bool]$env:SKIP_OLLAMA
$SkipSetup  = [bool]$env:SKIP_SETUP
$SkipModels = [bool]$env:SKIP_MODELS

# ============================================================================
# Helpers
# ============================================================================

function Write-Info { param([string]$Message) Write-Host "==> " -ForegroundColor Cyan -NoNewline; Write-Host $Message }
function Write-Warn { param([string]$Message) Write-Host "!   " -ForegroundColor Yellow -NoNewline; Write-Host $Message }
function Write-Ok   { param([string]$Message) Write-Host "OK  " -ForegroundColor Green -NoNewline; Write-Host $Message }
function Write-Err  { param([string]$Message) Write-Host "!!  " -ForegroundColor Red -NoNewline; Write-Host $Message }

function Write-H1 {
    param([string]$Title)
    Write-Host ""
    Write-Host $Title -ForegroundColor White
}

function Fail {
    param([string]$Message)
    Write-Err $Message
    exit 1
}

function Test-Command {
    param([string]$Name)
    $null -ne (Get-Command $Name -ErrorAction SilentlyContinue)
}

function Update-SessionPath {
    # Refresh the current session's PATH from the machine + user environment.
    # Needed after winget installs a new tool so Get-Command can find it.
    $machinePath = [System.Environment]::GetEnvironmentVariable('Path', 'Machine')
    $userPath    = [System.Environment]::GetEnvironmentVariable('Path', 'User')
    $env:Path = "$machinePath;$userPath"
}

# Resolve the full path to powershell.exe using $PSHOME so child process
# spawning works even when the session PATH doesn't include the PS directory.
# This is the most common cause of install failures on fresh machines.
function Get-PowerShellExe {
    $pshExe = Join-Path $PSHOME 'powershell.exe'
    if (Test-Path $pshExe) { return $pshExe }
    # Fallback: try to find it in known system locations
    $fallbacks = @(
        "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe",
        "$env:SystemRoot\SysWOW64\WindowsPowerShell\v1.0\powershell.exe"
    )
    foreach ($path in $fallbacks) {
        if (Test-Path $path) { return $path }
    }
    Fail "Cannot locate powershell.exe. Expected at '$pshExe'. Check your Windows installation."
}

# ============================================================================
# Header
# ============================================================================

Write-Host ""
Write-Host "+-----------------------------------------------------+" -ForegroundColor Cyan
Write-Host "|  Claudia Autonomous - Windows installer            |" -ForegroundColor Cyan
Write-Host "|  Trust-aware local AI memory + chief of staff agent|" -ForegroundColor Cyan
Write-Host "+-----------------------------------------------------+" -ForegroundColor Cyan
Write-Host ""
Write-Host "Target: " -NoNewline -ForegroundColor DarkGray
Write-Host $InstallDir
Write-Host "Branch: " -NoNewline -ForegroundColor DarkGray
Write-Host $Branch
Write-Host ""

# ============================================================================
# Environment checks
# ============================================================================

Write-H1 "Checking environment"

# PowerShell version
if ($PSVersionTable.PSVersion.Major -lt 5) {
    Fail "PowerShell 5.1 or newer required. Found: $($PSVersionTable.PSVersion)"
}
Write-Ok "PowerShell $($PSVersionTable.PSVersion)"

# OS check
$isWindowsOS = $false
if ($PSVersionTable.PSVersion.Major -ge 6) {
    $isWindowsOS = $IsWindows
} else {
    # Windows PowerShell 5.1 only runs on Windows
    $isWindowsOS = $true
}
if (-not $isWindowsOS) {
    Fail "install.ps1 is for Windows only. Use install.sh on macOS/Linux/WSL2."
}
Write-Ok "Windows detected"

# winget check (used to install prerequisites)
$hasWinget = Test-Command 'winget'
if ($hasWinget) {
    Write-Ok "winget: present"
} else {
    Write-Warn "winget not found. Prerequisite auto-install will not work."
    Write-Warn "winget ships with Windows 11 and recent Windows 10 builds. Update the App Installer from the Microsoft Store if missing."
}

# Git check
if (-not (Test-Command 'git')) {
    if ($hasWinget) {
        Write-Info "git not found. Installing via winget..."
        & winget install --silent --accept-package-agreements --accept-source-agreements --id Git.Git
        if ($LASTEXITCODE -ne 0) {
            Fail "winget install of Git.Git failed (exit code $LASTEXITCODE). Install manually: https://git-scm.com/download/win"
        }
        Update-SessionPath
        if (-not (Test-Command 'git')) {
            Fail "git was installed but is not on PATH in this session. Restart PowerShell and re-run the installer."
        }
        Write-Ok "git installed via winget"
    } else {
        Fail "git not found and winget is unavailable. Install git manually: https://git-scm.com/download/win"
    }
} else {
    $gitVer = (& git --version 2>$null).Trim()
    Write-Ok "$gitVer"
}

# Python 3.11+ check
$pythonOk = $false
foreach ($candidate in 'python', 'python3', 'py') {
    if (Test-Command $candidate) {
        try {
            $versionString = & $candidate --version 2>&1 | Out-String
            if ($versionString -match '(\d+)\.(\d+)\.(\d+)') {
                $major = [int]$Matches[1]
                $minor = [int]$Matches[2]
                if ($major -eq 3 -and $minor -ge 11) {
                    $pythonOk = $true
                    Write-Ok "Python $($Matches[0]) (via '$candidate')"
                    break
                }
            }
        } catch { }
    }
}

if (-not $pythonOk) {
    if ($hasWinget) {
        Write-Info "Python 3.11+ not found. Installing Python 3.11 via winget..."
        & winget install --silent --accept-package-agreements --accept-source-agreements --id Python.Python.3.11
        if ($LASTEXITCODE -ne 0) {
            Fail "winget install of Python.Python.3.11 failed (exit code $LASTEXITCODE). Install manually: https://www.python.org/downloads/"
        }
        Update-SessionPath
        if (Test-Command 'python') {
            Write-Ok "Python 3.11 installed via winget"
        } else {
            Write-Warn "Python was installed but is not on PATH in this session."
            Write-Warn "uv (installed by setup-claudia.ps1) can provision Python on its own, so this is not fatal."
        }
    } else {
        Write-Warn "Python 3.11+ not found and winget unavailable."
        Write-Warn "Install Python manually, OR let uv (installed next) provision Python on its own."
        Write-Warn "Python downloads: https://www.python.org/downloads/"
    }
}

# ============================================================================
# Install / check Ollama
# ============================================================================

if (-not $SkipOllama) {
    Write-H1 "Checking Ollama"
    if (Test-Command 'ollama') {
        Write-Ok "ollama: already installed"
    } else {
        if ($hasWinget) {
            Write-Info "Installing Ollama via winget..."
            & winget install --silent --accept-package-agreements --accept-source-agreements --id Ollama.Ollama
            if ($LASTEXITCODE -ne 0) {
                Write-Warn "winget install of Ollama.Ollama failed (exit code $LASTEXITCODE)"
                Write-Warn "Download manually: https://ollama.com/download/windows"
                Write-Warn "Then re-run the installer with `$env:SKIP_OLLAMA = '1' to skip this check"
                exit 1
            }
            Update-SessionPath
            if (Test-Command 'ollama') {
                Write-Ok "Ollama installed via winget"
            } else {
                Write-Warn "Ollama was installed but is not yet on PATH in this session."
                Write-Warn "After the installer finishes, restart PowerShell and run 'ollama pull all-minilm:l6-v2' and 'ollama pull qwen2.5:3b' manually."
                # Don't fail; the venv install can still succeed without Ollama being reachable yet
            }
        } else {
            Write-Warn "winget not available. Download Ollama manually: https://ollama.com/download/windows"
            Write-Warn "Then re-run the installer with `$env:SKIP_OLLAMA = '1' to skip this check"
            exit 1
        }
    }
}

# ============================================================================
# Clone or update the repo
# ============================================================================

Write-H1 "Fetching the code"

$gitDirMarker = Join-Path $InstallDir '.git'
if ((Test-Path $InstallDir) -and (Test-Path $gitDirMarker)) {
    # .git can be either a directory (normal clone) or a file (submodule pointer).
    # Test-Path returns true for both.
    Write-Info "Repo already exists at $InstallDir - updating..."
    $originalLocation = Get-Location
    try {
        Set-Location $InstallDir
        & git fetch --quiet origin $Branch
        if ($LASTEXITCODE -ne 0) { Fail "git fetch failed" }
        & git checkout --quiet $Branch
        if ($LASTEXITCODE -ne 0) { Fail "git checkout failed" }
        # Try a fast-forward pull; non-ff just warns
        & git pull --quiet --ff-only origin $Branch 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Ok "Updated to latest $Branch"
        } else {
            Write-Warn "Non-fast-forward - leaving local state alone."
            Write-Warn "To reset: Remove-Item -Recurse -Force '$InstallDir' and re-run installer"
        }
    } finally {
        Set-Location $originalLocation
    }
} elseif (Test-Path $InstallDir) {
    Fail "$InstallDir exists but is not a git repo. Move it aside and re-run."
} else {
    Write-Info "Cloning $RepoUrl to $InstallDir..."
    $parent = Split-Path $InstallDir -Parent
    if ($parent -and -not (Test-Path $parent)) {
        New-Item -Path $parent -ItemType Directory -Force | Out-Null
    }
    & git clone --quiet --branch $Branch --recurse-submodules $RepoUrl $InstallDir
    if ($LASTEXITCODE -ne 0) {
        Fail "git clone failed (exit code $LASTEXITCODE)"
    }
    Write-Ok "Cloned"
}

# ============================================================================
# Run setup-claudia.ps1
# ============================================================================

if (-not $SkipSetup) {
    Write-H1 "Running setup-claudia.ps1"
    $setupScript = Join-Path $InstallDir 'setup-claudia.ps1'
    if (-not (Test-Path $setupScript)) {
        Fail "setup-claudia.ps1 not found at $setupScript (did the clone succeed?)"
    }
    # Use $PSHOME to get the absolute path to powershell.exe so this works
    # even when powershell.exe is not in the active session PATH. This is the
    # most common install failure on fresh machines and restricted environments.
    $psExe = Get-PowerShellExe
    Write-Info "Launching setup in a child PowerShell process (ExecutionPolicy Bypass)..."
    Write-Info "Using: $psExe"
    & $psExe -NoProfile -ExecutionPolicy Bypass -File $setupScript -NonInteractive
    if ($LASTEXITCODE -ne 0) {
        Fail "setup-claudia.ps1 failed with exit code $LASTEXITCODE. Scroll up for details."
    }
    Write-Ok "setup-claudia.ps1 completed"

    # Refresh PATH in this session so claudia.cmd (added by setup) is reachable
    # without the user needing to restart PowerShell.
    Update-SessionPath
}

# ============================================================================
# Pull Ollama models for the memory plugin
# ============================================================================

if ((-not $SkipOllama) -and (-not $SkipModels) -and (Test-Command 'ollama')) {
    Write-H1 "Pulling Ollama models for Claudia Memory"
    # Check if the Ollama daemon is reachable
    $daemonRunning = $false
    try {
        $null = Invoke-WebRequest -Uri 'http://localhost:11434/api/tags' -UseBasicParsing -TimeoutSec 3 -ErrorAction Stop
        $daemonRunning = $true
    } catch {
        $daemonRunning = $false
    }

    if (-not $daemonRunning) {
        Write-Warn "Ollama daemon isn't running yet."
        Write-Warn "The Ollama installer usually starts a background service, but on first install you may need to"
        Write-Warn "launch Ollama from the Start menu, or run 'ollama serve' in a separate PowerShell window."
        Write-Warn ""
        Write-Warn "Then pull the two default models manually:"
        Write-Warn "    ollama pull all-minilm:l6-v2"
        Write-Warn "    ollama pull qwen2.5:3b"
    } else {
        Write-Info "Pulling all-minilm:l6-v2 (embeddings, ~23 MB)..."
        & ollama pull all-minilm:l6-v2
        if ($LASTEXITCODE -ne 0) { Write-Warn "Failed to pull all-minilm:l6-v2" }
        Write-Info "Pulling qwen2.5:3b (entity extraction + commitment detection, ~2 GB)..."
        & ollama pull qwen2.5:3b
        if ($LASTEXITCODE -ne 0) { Write-Warn "Failed to pull qwen2.5:3b" }
        Write-Ok "Models ready"
    }
}

# ============================================================================
# Done
# ============================================================================

Write-Host ""
Write-Host "+-----------------------------------------------------+" -ForegroundColor Green
Write-Host "|  OK  Claudia installed.                             |" -ForegroundColor Green
Write-Host "+-----------------------------------------------------+" -ForegroundColor Green
Write-Host ""

# Check if claudia is reachable in the current session after the PATH refresh.
# If it is, skip the restart step entirely - most users won't need it.
$claudiaAvailable = Test-Command 'claudia'

if ($claudiaAvailable) {
    Write-Host "The 'claudia' command is ready in this session." -ForegroundColor Green
    Write-Host "No restart needed." -ForegroundColor DarkGray
    Write-Host ""
} else {
    Write-Host "Almost ready." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  One more step: open a new PowerShell window." -ForegroundColor White
    Write-Host ""
    Write-Host "  Why? The installer added 'claudia' to your PATH, but that change" -ForegroundColor DarkGray
    Write-Host "  only takes effect in new terminal sessions, not this one." -ForegroundColor DarkGray
    Write-Host "  Just close this window and open a fresh PowerShell to continue." -ForegroundColor DarkGray
    Write-Host ""
}

Write-Host "Next steps:" -ForegroundColor White
Write-Host ""

$step = 1

if (-not $claudiaAvailable) {
    Write-Host "  $step. " -NoNewline
    Write-Host "Open a new PowerShell window" -ForegroundColor White
    Write-Host "     (closes this session, picks up the updated PATH)"
    Write-Host ""
    $step++
}

Write-Host "  $step. " -NoNewline
Write-Host "Activate Claudia's hybrid memory provider:" -ForegroundColor White
Write-Host "       claudia memory setup"
Write-Host ""
$step++

Write-Host "  $step. " -NoNewline
Write-Host "Verify everything is wired up:" -ForegroundColor White
Write-Host "       claudia doctor"
Write-Host ""
$step++

Write-Host "  $step. " -NoNewline
Write-Host "(Optional) Run the offline memory demo:" -ForegroundColor White
Write-Host "       cd $InstallDir"
Write-Host "       python -m plugins.memory.claudia.demo"
Write-Host ""
$step++

Write-Host "  $step. " -NoNewline
Write-Host "Start a session:" -ForegroundColor White
Write-Host "       claudia"
Write-Host ""

Write-Host "Docs: " -NoNewline -ForegroundColor DarkGray
Write-Host "https://github.com/kbanc85/claudia-autonomous"
Write-Host "Memory plugin: " -NoNewline -ForegroundColor DarkGray
Write-Host "$InstallDir\plugins\memory\claudia\README.md"
Write-Host ""
