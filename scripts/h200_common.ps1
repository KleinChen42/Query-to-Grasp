$Script:H200DefaultRemoteUser = "zetyun"
$Script:H200DefaultRemoteHost = "183.166.183.2"
$Script:H200DefaultRemotePort = "60071"
$Script:H200DefaultRemoteWorkspace = "/home/zetyun/OpenMythos_test"
$Script:H200DefaultKeyName = "hd03-tenant13-research-20260405"
$Script:H200DefaultSourceKey = "E:\CODE\KIWI\OpenMythos\hd03-tenant13-research-20260405"
$Script:H200DefaultSshExe = "C:\Windows\System32\OpenSSH\ssh.exe"
$Script:H200DefaultScpExe = "C:\Windows\System32\OpenSSH\scp.exe"

function Get-H200RepoRoot {
    param(
        [string]$WorkspaceRoot = ""
    )

    if ($WorkspaceRoot) {
        return [System.IO.Path]::GetFullPath($WorkspaceRoot)
    }

    return [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
}

function Get-H200AskPassPath {
    param(
        [string]$WorkspaceRoot
    )

    $workspace = Get-H200RepoRoot -WorkspaceRoot $WorkspaceRoot
    return Join-Path $workspace "ssh_askpass.cmd"
}

function Join-H200RemotePath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Base,
        [Parameter(Mandatory = $true)]
        [string]$Child
    )

    $cleanBase = $Base.TrimEnd("/")
    $cleanChild = $Child.Replace("\", "/").Trim()
    $cleanChild = $cleanChild.TrimStart(".")
    $cleanChild = $cleanChild.TrimStart("/")
    if (-not $cleanChild) {
        return $cleanBase
    }
    return "$cleanBase/$cleanChild"
}

function Get-H200RemoteParent {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    $normalized = $Path.Replace("\", "/").TrimEnd("/")
    $lastSlash = $normalized.LastIndexOf("/")
    if ($lastSlash -lt 1) {
        return "/"
    }
    return $normalized.Substring(0, $lastSlash)
}

function Resolve-H200LocalPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$WorkspaceRoot,
        [Parameter(Mandatory = $true)]
        [string]$RelativeOrAbsolutePath
    )

    if ([System.IO.Path]::IsPathRooted($RelativeOrAbsolutePath)) {
        return [System.IO.Path]::GetFullPath($RelativeOrAbsolutePath)
    }

    return [System.IO.Path]::GetFullPath((Join-Path $WorkspaceRoot $RelativeOrAbsolutePath))
}

function Resolve-H200RemotePath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RemoteWorkspace,
        [Parameter(Mandatory = $true)]
        [string]$RelativeOrAbsolutePath
    )

    $candidate = $RelativeOrAbsolutePath.Replace("\", "/").Trim()
    if ($candidate.StartsWith("/")) {
        return $candidate
    }

    return Join-H200RemotePath -Base $RemoteWorkspace -Child $candidate
}

function Initialize-H200Context {
    param(
        [string]$WorkspaceRoot = "",
        [string]$RemoteUser = $Script:H200DefaultRemoteUser,
        [string]$RemoteHost = $Script:H200DefaultRemoteHost,
        [string]$RemotePort = $Script:H200DefaultRemotePort,
        [string]$RemoteWorkspace = $Script:H200DefaultRemoteWorkspace,
        [string]$KeyName = $Script:H200DefaultKeyName,
        [string]$SourceKey = $Script:H200DefaultSourceKey,
        [string]$SshExe = $Script:H200DefaultSshExe,
        [string]$ScpExe = $Script:H200DefaultScpExe
    )

    $workspace = Get-H200RepoRoot -WorkspaceRoot $WorkspaceRoot
    $sshDir = Join-Path $HOME ".ssh"
    $keyPath = Join-Path $sshDir $KeyName
    $askPass = Get-H200AskPassPath -WorkspaceRoot $workspace

    if (!(Test-Path -LiteralPath $keyPath)) {
        if (!(Test-Path -LiteralPath $SourceKey)) {
            throw "SSH key not found at $keyPath or source key path $SourceKey."
        }

        New-Item -ItemType Directory -Force -Path $sshDir | Out-Null
        Copy-Item -LiteralPath $SourceKey -Destination $keyPath -Force
        icacls $keyPath /inheritance:r /grant:r "$env:USERNAME`:R" | Out-Null
    }

    if (!(Test-Path -LiteralPath $askPass)) {
        throw "ssh_askpass.cmd not found at $askPass."
    }

    if (!$env:SSH_KEY_PASSPHRASE) {
        throw "SSH_KEY_PASSPHRASE must be set before running H200 sync helpers."
    }

    $env:SSH_ASKPASS = $askPass
    $env:SSH_ASKPASS_REQUIRE = "force"
    $env:DISPLAY = ":0"

    $commonOpts = @(
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "UserKnownHostsFile=NUL",
        "-o", "LogLevel=ERROR",
        "-i", $keyPath
    )

    return @{
        WorkspaceRoot = $workspace
        RemoteUser = $RemoteUser
        RemoteHost = $RemoteHost
        RemotePort = $RemotePort
        RemoteWorkspace = $RemoteWorkspace
        SshExe = $SshExe
        ScpExe = $ScpExe
        SshArgsBase = $commonOpts + @("-p", $RemotePort, "$RemoteUser@$RemoteHost")
        ScpArgsBase = $commonOpts + @("-P", $RemotePort)
    }
}
