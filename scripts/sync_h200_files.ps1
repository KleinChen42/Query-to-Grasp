param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("push", "pull")]
    [string]$Direction,
    [Parameter(Mandatory = $true)]
    [string[]]$Paths,
    [string]$WorkspaceRoot = "",
    [string]$RemoteWorkspace = $null,
    [switch]$Recursive,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "h200_common.ps1")

$ctx = Initialize-H200Context -WorkspaceRoot $WorkspaceRoot
$resolvedRemoteWorkspace = if ($null -ne $RemoteWorkspace -and $RemoteWorkspace -ne "") {
    $RemoteWorkspace
} else {
    $ctx.RemoteWorkspace
}

$normalizedPaths = @()
foreach ($rawPath in $Paths) {
    foreach ($candidate in ($rawPath -split ",")) {
        $trimmed = $candidate.Trim()
        if ($trimmed) {
            $normalizedPaths += $trimmed
        }
    }
}

foreach ($path in $normalizedPaths) {
    $localPath = Resolve-H200LocalPath -WorkspaceRoot $ctx.WorkspaceRoot -RelativeOrAbsolutePath $path
    $remotePath = Resolve-H200RemotePath -RemoteWorkspace $resolvedRemoteWorkspace -RelativeOrAbsolutePath $path
    $remoteSpec = "$($ctx.RemoteUser)@$($ctx.RemoteHost):$remotePath"
    $localParent = Split-Path -Parent $localPath
    $remoteParent = Get-H200RemoteParent -Path $remotePath

    if ($Direction -eq "push") {
        if (!(Test-Path -LiteralPath $localPath)) {
            throw "Local path does not exist: $localPath"
        }

        $isDirectory = Test-Path -LiteralPath $localPath -PathType Container
        $mkdirCommand = "mkdir -p $remoteParent"
        Write-Host "[h200 push] $path"
        if ($DryRun) {
            Write-Host "  mkdir: $mkdirCommand"
            Write-Host "  copy : $localPath -> $remoteSpec"
            continue
        }

        & $ctx.SshExe @($ctx.SshArgsBase + @($mkdirCommand))
        if ($LASTEXITCODE -ne 0) {
            throw "Remote mkdir failed for $remoteParent ($LASTEXITCODE)."
        }

        if ($isDirectory -or $Recursive) {
            & $ctx.ScpExe @($ctx.ScpArgsBase + @("-r", $localPath, "$($ctx.RemoteUser)@$($ctx.RemoteHost):$remoteParent"))
        } else {
            & $ctx.ScpExe @($ctx.ScpArgsBase + @($localPath, $remoteSpec))
        }
        if ($LASTEXITCODE -ne 0) {
            throw "scp push failed for $path ($LASTEXITCODE)."
        }
        continue
    }

    Write-Host "[h200 pull] $path"
    if ($DryRun) {
        Write-Host "  copy : $remoteSpec -> $localPath"
        continue
    }

    New-Item -ItemType Directory -Force -Path $localParent | Out-Null
    if ($Recursive) {
        & $ctx.ScpExe @($ctx.ScpArgsBase + @("-r", $remoteSpec, $localParent))
    } else {
        & $ctx.ScpExe @($ctx.ScpArgsBase + @($remoteSpec, $localPath))
    }
    if ($LASTEXITCODE -ne 0) {
        throw "scp pull failed for $path ($LASTEXITCODE)."
    }
}
