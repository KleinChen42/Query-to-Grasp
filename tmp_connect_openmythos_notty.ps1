param(
    [Parameter(Mandatory = $true)]
    [string]$RemoteCommand
)

$ErrorActionPreference = "Stop"

$SourceKey = "E:\CODE\KIWI\OpenMythos\hd03-tenant13-research-20260405"
$SshDir = Join-Path $HOME ".ssh"
$KeyPath = Join-Path $SshDir "hd03-tenant13-research-20260405"
$AskPass = "E:\CODE\KIWI\OpenMythos\tools\ssh_askpass.cmd"
$SshExe = "C:\Windows\System32\OpenSSH\ssh.exe"

$RemoteUser = "zetyun"
$RemoteHost = "183.166.183.2"
$RemotePort = "60071"

if (!(Test-Path -LiteralPath $KeyPath)) {
    New-Item -ItemType Directory -Force -Path $SshDir | Out-Null
    Copy-Item -LiteralPath $SourceKey -Destination $KeyPath -Force
    icacls $KeyPath /inheritance:r /grant:r "$env:USERNAME`:R" | Out-Null
}

if (!$env:SSH_KEY_PASSPHRASE) {
    throw "SSH_KEY_PASSPHRASE must be set before running this script."
}

$env:SSH_ASKPASS = $AskPass
$env:SSH_ASKPASS_REQUIRE = "force"
$env:DISPLAY = ":0"

$SshArgs = @(
    "-o", "StrictHostKeyChecking=accept-new",
    "-o", "UserKnownHostsFile=NUL",
    "-i", $KeyPath,
    "-p", $RemotePort,
    "$RemoteUser@$RemoteHost",
    $RemoteCommand
)

& $SshExe @SshArgs
