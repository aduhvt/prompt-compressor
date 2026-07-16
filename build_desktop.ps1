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
  --copy-metadata sentence-transformers `
  desktop_gui.py

Write-Host ""
Write-Host "Build complete: $projectRoot\dist\PromptCompressor.exe"
Write-Host "Copy that file to your desktop or create a shortcut to it."
