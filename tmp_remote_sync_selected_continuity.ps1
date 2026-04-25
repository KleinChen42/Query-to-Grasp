$ErrorActionPreference = "Stop"

$Workspace = "E:\CODE\Query-to-Grasp"
$SourceKey = "E:\CODE\KIWI\OpenMythos\hd03-tenant13-research-20260405"
$SshDir = Join-Path $HOME ".ssh"
$KeyPath = Join-Path $SshDir "hd03-tenant13-research-20260405"
$AskPass = "E:\CODE\KIWI\OpenMythos\tools\ssh_askpass.cmd"
$SshExe = "C:\Windows\System32\OpenSSH\ssh.exe"
$ScpExe = "C:\Windows\System32\OpenSSH\scp.exe"

$RemoteUser = "zetyun"
$RemoteHost = "183.166.183.2"
$RemotePort = "60071"
$RemoteBase = "/home/zetyun/OpenMythos_test"

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

$CommonOpts = @(
    "-o", "StrictHostKeyChecking=accept-new",
    "-o", "UserKnownHostsFile=NUL",
    "-i", $KeyPath
)

$ScpArgsBase = $CommonOpts + @("-P", $RemotePort)
$SshArgsBase = $CommonOpts + @("-p", $RemotePort, "$RemoteUser@$RemoteHost")

$Files = @(
    "scripts/run_multiview_fusion_debug.py",
    "scripts/run_multiview_fusion_benchmark.py",
    "scripts/generate_reobserve_policy_report.py",
    "src/memory/object_memory_3d.py",
    "src/policy/target_selector.py",
    "tests/test_memory.py",
    "tests/test_target_selector.py",
    "tests/test_run_multiview_fusion_debug.py",
    "tests/test_run_multiview_fusion_benchmark.py",
    "tests/test_generate_reobserve_policy_report.py"
)

& $SshExe @SshArgsBase "mkdir -p $RemoteBase/scripts $RemoteBase/src/memory $RemoteBase/src/policy $RemoteBase/tests"
if ($LASTEXITCODE -ne 0) { throw "remote mkdir failed ($LASTEXITCODE)" }

foreach ($file in $Files) {
    $local = Join-Path $Workspace $file
    $remote = "${RemoteUser}@${RemoteHost}:$RemoteBase/$file"
    Write-Host "Uploading $file"
    & $ScpExe @ScpArgsBase $local $remote
    if ($LASTEXITCODE -ne 0) { throw "scp $file failed ($LASTEXITCODE)" }
}

Write-Host "Remote continuity patch sync complete."
