# End-to-end pipeline: download -> split -> vocab -> features -> train -> evaluate.
# Run this on a machine with real internet access (Kaggle download, optional
# Hugging Face upload) and, ideally, a GPU.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\run_pipeline.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\run_pipeline.ps1 -LocalDir "C:\path\to\already-extracted-dataset"

param(
    [string]$LocalDir = ""
)

$ErrorActionPreference = "Stop"

# cd to project root (parent of this script's folder)
Set-Location (Join-Path $PSScriptRoot "..")

function Run-Step($label, $script) {
    Write-Host "== $label ==" -ForegroundColor Cyan
    python $script
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Step failed: $script" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

if ($LocalDir -ne "") {
    Write-Host "== 1/6 Using local dataset directory: $LocalDir ==" -ForegroundColor Cyan
    python scripts\download_data.py --local-dir "$LocalDir"
    if ($LASTEXITCODE -ne 0) { Write-Host "Step failed: download_data.py --local-dir" -ForegroundColor Red; exit $LASTEXITCODE }
} else {
    Run-Step "1/6 Downloading Flickr8k from Kaggle"    "scripts\download_data.py"
}
Run-Step "2/6 Building leakage-free train/val/test split" "scripts\split_dataset.py"
Run-Step "3/6 Building vocabulary (training split only)"  "scripts\build_vocab.py"
Run-Step "4/6 Extracting and caching CNN features"        "scripts\extract_features.py"
Run-Step "5/6 Training the caption model"                 "scripts\train.py"
Run-Step "6/6 Evaluating on the held-out test set"         "scripts\evaluate.py"

Write-Host "Pipeline complete. See artifacts\results\ for metrics and qualitative examples." -ForegroundColor Green
Write-Host "Optional: python scripts\upload_to_hub.py --repo-id <your-username>/flickr8k-caption-gen"
