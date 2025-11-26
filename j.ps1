param(
    [int]$Port = 8502
)
$scriptDir = Split-Path -LiteralPath $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $scriptDir
# Activate virtual environment if present
$activate = Join-Path $scriptDir 'venv\Scripts\Activate.ps1'
if (Test-Path $activate) {
    & $activate
}
# Run Streamlit using python -m to avoid exe launcher issues
python -m streamlit run .\app.py --server.port $Port --server.headless true
