param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('smoke', 'analyze', 'package', 'all')]
    [string]$Stage,

    [Parameter(Mandatory = $true)]
    [string]$InputRoot,

    [string]$PythonExe = 'C:\Users\Public\miniconda3\envs\opencode_env\python.exe'
)

$ErrorActionPreference = 'Stop'
$repository = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$root = (Resolve-Path $InputRoot).Path
if (-not (Test-Path -LiteralPath $PythonExe -PathType Leaf)) {
    throw "Python executable not found: $PythonExe"
}

function Invoke-Analysis([string[]]$Arguments) {
    & $PythonExe -B -m five_dataset_features.analyze @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Analysis command failed with exit code $LASTEXITCODE`: $($Arguments -join ' ')"
    }
}

Push-Location $repository
try {
    if ($Stage -in @('smoke', 'all')) {
        & $PythonExe -B -m five_dataset_features.smoke --root $root
        if ($LASTEXITCODE -ne 0) {
            throw "CUB smoke validation failed with exit code $LASTEXITCODE"
        }
    }
    if ($Stage -in @('analyze', 'all')) {
        foreach ($dataset in @('cotton', 'soyageing', 'soyglobal', 'cub', 'pets')) {
            Invoke-Analysis -Arguments @('dataset', '--root', $root, '--dataset', $dataset)
        }
        Invoke-Analysis -Arguments @('report', '--root', $root)
    }
    if ($Stage -in @('package', 'all')) {
        Invoke-Analysis -Arguments @('package', '--root', $root)
    }
} finally {
    Pop-Location
}
