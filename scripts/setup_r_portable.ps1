# Setup R-Portable for PassiAgent
# Downloads portable R to the project directory and configures rpy2.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts/setup_r_portable.ps1
#   powershell -ExecutionPolicy Bypass -File scripts/setup_r_portable.ps1 -RVersion "4.4.1"

param(
    [string]$RVersion = "4.4.1",
    [string]$ProjectRoot = $PSScriptRoot + "\.."
)

$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path $ProjectRoot).Path
$RPortableDir = Join-Path $ProjectRoot "R-Portable"
$RHome = Join-Path $RPortableDir "App\R-Portable"
$RBin = Join-Path $RHome "bin"

Write-Host "=== PassiAgent R-Portable Setup ===" -ForegroundColor Cyan
Write-Host "  Project: $ProjectRoot"
Write-Host "  R Version: $RVersion"
Write-Host "  Target: $RPortableDir"

# ── Check if already installed ──
if ($env:PASSI_R_HOME) {
    $existing = $env:PASSI_R_HOME
    Write-Host "`n[INFO] PASSI_R_HOME already set to: $existing" -ForegroundColor Yellow
}
if (Test-Path (Join-Path $RBin "R.exe")) {
    Write-Host "[INFO] R-Portable already exists at: $RHome" -ForegroundColor Green
    Write-Host "[INFO] R version: $(& (Join-Path $RBin 'R.exe') --version | Select-Object -First 1)"
    $configureOnly = $true
}
else {
    $configureOnly = $false
}

# ── Download and extract R-Portable ──
if (-not $configureOnly) {
    $ZipUrl = "https://sourceforge.net/projects/rportable/files/R-Portable/R-$RVersion/R-Portable-$RVersion.zip/download"
    $ZipFile = Join-Path $env:TEMP "R-Portable-$RVersion.zip"

    Write-Host "`n[1/3] Downloading R-Portable $RVersion ..."
    Write-Host "  URL: $ZipUrl"

    try {
        # Try SourceForge first
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        Invoke-WebRequest -Uri $ZipUrl -OutFile $ZipFile -UseBasicParsing
    }
    catch {
        Write-Host "[WARN] SourceForge download failed, trying CRAN mirror..." -ForegroundColor Yellow
        # Fallback: download R installer and extract
        $CranUrl = "https://cran.r-project.org/bin/windows/base/old/$RVersion/R-$RVersion-win.exe"
        Write-Host "  URL: $CranUrl"
        Write-Host "[ERROR] R-Portable download failed. Please download manually from:"
        Write-Host "  https://sourceforge.net/projects/rportable/"
        Write-Host "  Extract to: $RPortableDir"
        throw "Automatic download failed — manual setup required."
    }

    Write-Host "[2/3] Extracting R-Portable to $RPortableDir ..."
    Expand-Archive -Path $ZipFile -DestinationPath $RPortableDir -Force
    Remove-Item $ZipFile

    # Verify extraction
    if (-not (Test-Path (Join-Path $RBin "R.exe"))) {
        # Try alternative structure (R-Portable may extract differently)
        $altRBin = Get-ChildItem -Path $RPortableDir -Recurse -Filter "R.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($altRBin) {
            $global:RHome = $altRBin.Directory.Parent.FullName
            $global:RBin = $altRBin.Directory.FullName
        }
        else {
            Write-Host "[ERROR] R.exe not found after extraction. Archive structure may have changed."
            Write-Host "  Contents of $RPortableDir :"
            Get-ChildItem $RPortableDir -Recurse -Depth 2 | ForEach-Object { Write-Host "    $_" }
            throw "R.exe not found."
        }
    }

    Write-Host "[INFO] R-Portable extracted successfully." -ForegroundColor Green
}

# ── Verify R installation ──
Write-Host "`n[3/3] Verifying R installation ..."
$RExe = Join-Path $RBin "R.exe"
$RVersionOutput = & $RExe --version 2>&1 | Select-Object -First 1
Write-Host "  $RVersionOutput"

# ── Check essential packages ──
Write-Host "`nChecking Bioconductor packages..."
$checkScript = @"
cat("R_HOME:", R.home(), "\n")
cat(".libPaths():", .libPaths(), "\n")
pkgs <- c("DESeq2", "edgeR", "limma", "clusterProfiler", "WGCNA", "mixOmics", "MOFA2", "survival", "DSS", "DiffBind", "SNFtool", "fgsea")
for (p in pkgs) {
  if (requireNamespace(p, quietly = TRUE)) {
    cat(sprintf("  [OK] %s\n", p))
  } else {
    cat(sprintf("  [MISSING] %s\n", p))
  }
}
"@
& $RExe --no-save -e $checkScript 2>&1

# ── Generate .env config ──
$envFile = Join-Path $ProjectRoot ".env"
$envContent = @"
# PassiAgent R Configuration
PASSI_EXECUTION__R_HOME=$($RHome -replace '\\','\\')
PASSI_EXECUTION__R_PATH=$($RBin -replace '\\','\\')\Rscript.exe
PASSI_EXECUTION__RPY2_ENABLED=true

# Add R to PATH for rpy2 (important on Windows)
# The R.dll must be findable — rpy2 uses R_HOME/bin/<arch>/R.dll
# Set R_LIBS_USER if you want a custom library path
PASSI_EXECUTION__R_LIB_PATH=
"@

# Only write if .env doesn't exist
if (-not (Test-Path $envFile)) {
    Set-Content -Path $envFile -Value $envContent -Encoding UTF8
    Write-Host "`n[OK] Created .env with R configuration" -ForegroundColor Green
}
else {
    Write-Host "`n[INFO] .env already exists — add these lines manually:" -ForegroundColor Yellow
    Write-Host $envContent
}

Write-Host "`n=== Setup Complete ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "R-Portable location: $RHome"
Write-Host "R binary: $RExe"
Write-Host ""
Write-Host "Verify with:"
Write-Host "  python -c ""import os; os.environ['R_HOME']='$RHome'; import rpy2.robjects; print('rpy2 OK')"""
