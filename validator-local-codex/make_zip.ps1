$ErrorActionPreference = "Stop"

# Build a clean validator-local.zip next to this script.
# Excludes runtime/cache files even if present locally.

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$out  = Join-Path (Split-Path -Parent $root) "validator-local.zip"

if (Test-Path $out) { Remove-Item $out }

$exclude = @(
  "validator-local\.venv\*",
  "validator-local\__pycache__\*",
  "validator-local\*.pyc",
  "validator-local\server.log",
  "validator-local\data\*"
)

Push-Location (Split-Path -Parent $root)
try {
  Compress-Archive -Path "validator-local\*" -DestinationPath $out -Force
  # Note: Compress-Archive can't exclude patterns, so we delete unwanted entries by re-creating.
  $tmp = Join-Path (Split-Path -Parent $root) "validator-local.tmp"
  if (Test-Path $tmp) { Remove-Item $tmp -Recurse -Force }
  New-Item -ItemType Directory -Path $tmp | Out-Null
  Copy-Item -Recurse -Force "validator-local" $tmp
  foreach ($pat in $exclude) {
    Get-ChildItem -LiteralPath $tmp -Recurse -Force | Where-Object { $_.FullName -like (Join-Path $tmp $pat) } | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
  }
  if (Test-Path $out) { Remove-Item $out }
  Compress-Archive -Path (Join-Path $tmp "validator-local\*") -DestinationPath $out -Force
  Remove-Item $tmp -Recurse -Force
  Write-Host "Wrote: $out"
}
finally {
  Pop-Location
}
