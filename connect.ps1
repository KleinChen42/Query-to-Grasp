param(
    [string]$RemoteDir     = "",
    [string]$RemoteCommand = "",
    [switch]$NoCD
)

# ── 服务器配置（改这里就够了）──────────────────────────────────
$RemoteUser = "zetyun"
$RemoteHost = "183.166.183.2"
$RemotePort = "60071"
$SshKeyName = "hd03-tenant13-research-20260405"
# ──────────────────────────────────────────────────────────────

$ErrorActionPreference = "Stop"
$AskPassCmd = Join-Path $PSScriptRoot "ssh_askpass.cmd"
$KeyPath    = Join-Path $HOME ".ssh\$SshKeyName"

# 找 ssh.exe
$SshExe = "C:\Windows\System32\OpenSSH\ssh.exe"
if (!(Test-Path $SshExe)) {
    $found = Get-Command ssh -ErrorAction SilentlyContinue
    if ($found) { $SshExe = $found.Source }
    else { throw "未找到 ssh.exe，请启用 Windows 可选功能 OpenSSH Client" }
}

# 私钥检查
if (!(Test-Path $KeyPath)) {
    Write-Host "[warn] SSH key not found: $KeyPath" -ForegroundColor Yellow
    Write-Host "       Copy the key file '$SshKeyName' to that path, or edit `$SshKeyName in this script."
    exit 1
}
icacls $KeyPath /inheritance:r /grant:r "$env:USERNAME`:R" 2>&1 | Out-Null

# Passphrase（优先环境变量，再提示输入）
if (!$env:SSH_KEY_PASSPHRASE) {
    $Secure = Read-Host "SSH key passphrase (blank = none)" -AsSecureString
    $Bstr   = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($Secure)
    try   { $env:SSH_KEY_PASSPHRASE = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($Bstr) }
    finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($Bstr) }
}

# 让 ssh 无提示读取 passphrase
$env:SSH_ASKPASS         = $AskPassCmd
$env:SSH_ASKPASS_REQUIRE = "force"
$env:DISPLAY             = ":0"

# 基础 ssh 参数
$SshArgs = @(
    "-t",
    "-o", "StrictHostKeyChecking=accept-new",
    "-o", "ServerAliveInterval=60",
    "-o", "ServerAliveCountMax=10",
    "-i", $KeyPath,
    "-p", $RemotePort,
    "$RemoteUser@$RemoteHost"
)

# 远端动作
if ($RemoteCommand) {
    $SshArgs += $RemoteCommand
} elseif ((-not $NoCD) -and $RemoteDir) {
    $SshArgs += "cd $RemoteDir && exec bash --login"
} else {
    $SshArgs += "exec bash --login"
}

$dest = if ($RemoteDir) { "$RemoteDir" } else { "~" }
Write-Host "[h200] $RemoteUser@${RemoteHost}:$RemotePort  dir=$dest" -ForegroundColor Cyan
& $SshExe @SshArgs
