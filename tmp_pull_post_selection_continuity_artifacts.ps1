$ErrorActionPreference = "Stop"

$LocalBase = "E:\CODE\Query-to-Grasp\outputs\h200_60071_post_selection_continuity_ambiguity_compact_seed0"
$SourceKey = "E:\CODE\KIWI\OpenMythos\hd03-tenant13-research-20260405"
$SshDir = Join-Path $HOME ".ssh"
$KeyPath = Join-Path $SshDir "hd03-tenant13-research-20260405"
$AskPass = "E:\CODE\KIWI\OpenMythos\tools\ssh_askpass.cmd"
$ScpExe = "C:\Windows\System32\OpenSSH\scp.exe"

$RemoteUser = "zetyun"
$RemoteHost = "183.166.183.2"
$RemotePort = "60071"
$RemoteBase = "/home/zetyun/OpenMythos_test/outputs/h200_60071_post_selection_continuity_ambiguity_compact_seed0"

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
    "-i", $KeyPath,
    "-P", $RemotePort
)

$Files = @(
    "ambiguity_compact_hf_no_clip/benchmark_summary.json",
    "ambiguity_compact_hf_no_clip/benchmark_rows.json",
    "ambiguity_compact_hf_with_clip_notty/benchmark_summary.json",
    "ambiguity_compact_hf_with_clip_notty/benchmark_rows.json",
    "reobserve_policy_report_post_selection_continuity.md",
    "reobserve_policy_report_post_selection_continuity.json"
)

foreach ($file in $Files) {
    $target = Join-Path $LocalBase $file
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $target) | Out-Null
    $remote = "${RemoteUser}@${RemoteHost}:$RemoteBase/$file"
    Write-Host "Downloading $file"
    & $ScpExe @CommonOpts $remote $target
    if ($LASTEXITCODE -ne 0) { throw "scp $file failed ($LASTEXITCODE)" }
}

Write-Host "Remote post-selection continuity artifacts pulled locally."
