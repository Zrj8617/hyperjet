$ErrorActionPreference = "Stop"

$repositoryRoot = Split-Path -Parent $PSScriptRoot
$reportDirectory = Join-Path $repositoryRoot "docs\superpowers\reports"
$jsonPath = Join-Path $reportDirectory "2026-09-07-reward-redesign-0700-status.json"
$markdownPath = Join-Path $reportDirectory "2026-09-07-reward-redesign-0700-status.md"
$checkedAt = Get-Date

$runSpecs = @(
    "20260906_B1_seed0:78204", "20260906_B1_seed1:78205", "20260906_B1_seed2:78206",
    "20260906_D2_seed0:78207", "20260906_D2_seed1:78208", "20260906_D2_seed2:78209",
    "20260906_N0_seed0:78210", "20260906_N0_seed1:78211", "20260906_N0_seed2:78212",
    "20260906_A2_seed0:78213", "20260906_A2_seed1:78214", "20260906_A2_seed2:78215",
    "20260906_B2_seed0:78216", "20260906_B2_seed1:78217", "20260906_B2_seed2:78218",
    "20260906_B2D_seed0:78219", "20260906_B2D_seed1:78220", "20260906_B2D_seed2:78221",
    "20260906_D1_seed0:78222", "20260906_D1_seed1:78223", "20260906_D1_seed2:78224"
)
$specText = $runSpecs -join " "
$remoteCommand = @'
root=/data2/zrj2025/uav-results/audits
for spec in __RUN_SPECS__; do
    run=${spec%%:*}
    pid=${spec##*:}
    result="$root/$run/result.json"
    if [ -s "$result" ] && grep -q '"status": "completed"' "$result"; then
        status=completed
    elif kill -0 "$pid" 2>/dev/null; then
        status=running
    elif [ -s "$result" ]; then
        status=result_not_completed
    else
        status=missing_or_failed
    fi
    printf '%s|%s|%s|%s\n' "$run" "$pid" "$status" "$result"
done
'@.Replace("__RUN_SPECS__", $specText)

$sshOutput = @(& ssh -o BatchMode=yes -o ConnectTimeout=20 10.12.54.24 $remoteCommand 2>&1)
$rows = @()
foreach ($line in $sshOutput) {
    $parts = [string]$line -split "\|", 4
    if ($parts.Count -ne 4 -or $parts[0] -notmatch '^20260906_') {
        continue
    }
    $rows += [ordered]@{
        run = $parts[0]
        pid = [int]$parts[1]
        status = $parts[2]
        result_path = $parts[3]
    }
}

$completed = @($rows | Where-Object { $_.status -eq "completed" }).Count
$running = @($rows | Where-Object { $_.status -eq "running" }).Count
$problems = @($rows | Where-Object { $_.status -notin @("completed", "running") }).Count
$summaryStatus = if ($rows.Count -ne 21) {
    "check_failed"
} elseif ($completed -eq 21) {
    "all_completed"
} elseif ($problems -gt 0) {
    "attention_required"
} else {
    "still_running"
}

$payload = [ordered]@{
    schema = "reward_redesign_scheduled_completion_check_v1"
    checked_at_asia_shanghai = $checkedAt.ToString("o")
    status = $summaryStatus
    expected_run_count = 21
    observed_run_count = $rows.Count
    completed_count = $completed
    running_count = $running
    problem_count = $problems
    runs = $rows
    raw_ssh_output = if ($rows.Count -eq 21) { @() } else { $sshOutput }
}
$payload | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = @(
    "# Reward redesign scheduled completion check — 2026-09-07 07:00",
    "",
    "- Checked at: $($checkedAt.ToString('o'))",
    "- Status: **$summaryStatus**",
    "- Completed: $completed / 21",
    "- Still running: $running",
    "- Missing/failed/non-completed result: $problems",
    "",
    "| Run | PID | Status | Result |",
    "|---|---:|---|---|"
)
foreach ($row in $rows) {
    $lines += "| ``$($row.run)`` | $($row.pid) | $($row.status) | ``$($row.result_path)`` |"
}
$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8
