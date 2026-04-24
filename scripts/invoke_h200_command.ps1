param(
    [Parameter(Mandatory = $true)]
    [string]$RemoteCommand,
    [string]$WorkspaceRoot = "",
    [string]$RemoteWorkspace = $null,
    [switch]$NoWorkspacePrefix,
    [switch]$Tty
)

$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "h200_common.ps1")

$ctx = Initialize-H200Context -WorkspaceRoot $WorkspaceRoot
$resolvedRemoteWorkspace = if ($null -ne $RemoteWorkspace -and $RemoteWorkspace -ne "") {
    $RemoteWorkspace
} else {
    $ctx.RemoteWorkspace
}

$fullCommand = if ($NoWorkspacePrefix) {
    $RemoteCommand
} else {
    "cd $resolvedRemoteWorkspace && $RemoteCommand"
}

$sshArgs = @()
if ($Tty) {
    $sshArgs += "-t"
}
$sshArgs += $ctx.SshArgsBase + @($fullCommand)

Write-Host "[h200] $fullCommand"
& $ctx.SshExe @sshArgs
if ($LASTEXITCODE -ne 0) {
    throw "Remote command failed with exit code $LASTEXITCODE."
}
