$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Workspace = Split-Path -Parent $Root
$Venv = Join-Path $Workspace ".venv_ifnet"

if (!(Test-Path $Venv)) {
    python -m venv --system-site-packages $Venv
}

& (Join-Path $Venv "Scripts\python.exe") -m pip install --upgrade pip
& (Join-Path $Venv "Scripts\python.exe") -m pip install -e $Root
& (Join-Path $Venv "Scripts\python.exe") -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
