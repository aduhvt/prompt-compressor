$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

$python = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Virtual environment Python not found at $python"
}

& $python -m PyInstaller `
  --noconfirm `
  --clean `
  --onefile `
  --windowed `
  --name "PromptCompressor" `
  --add-data "compressor;compressor" `
  --collect-all customtkinter `
  --exclude-module torch `
  --exclude-module numpy `
  --exclude-module sentence_transformers `
  --exclude-module transformers `
  --exclude-module scipy `
  --exclude-module pandas `
  --exclude-module pyarrow `
  desktop_gui.py

Write-Host ""
Write-Host "Build complete: $projectRoot\dist\PromptCompressor.exe"
Write-Host "Copy that file to your desktop or create a shortcut to it."
