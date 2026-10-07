param(
  [Parameter(Mandatory = $true)]
  [string]$Video,

  [Parameter(Mandatory = $true)]
  [string]$RoiJson,

  [string]$Checkpoint,
  [string]$Output,
  [int]$FrameStep = 15,
  [int]$RoiPadding = 0,
  [double]$ScoreThreshold = 0.30,
  [Nullable[double]]$StartSeconds,
  [Nullable[double]]$DurationSeconds,
  [switch]$Overlay,
  [string]$Python = "$HOME\miniconda3\envs\python_env\python.exe"
)

$projectRoot = Split-Path $PSScriptRoot -Parent
$pipeline = Join-Path $PSScriptRoot "pipeline.py"
if (-not $Checkpoint) {
  $Checkpoint = Join-Path $PSScriptRoot "checkpoints\ant_detector_mobilenet_round2_reset_best.pt"
}
if (-not $Output) {
  $stem = [IO.Path]::GetFileNameWithoutExtension($Video)
  $Output = Join-Path $PSScriptRoot "outputs\${stem}_gpu"
}

$arguments = @(
  $pipeline,
  "track-video",
  "--video", $Video,
  "--roi-json", $RoiJson,
  "--checkpoint", $Checkpoint,
  "--output", $Output,
  "--device", "cuda",
  "--amp", "fp16",
  "--frame-step", $FrameStep,
  "--frame-batch-size", "2",
  "--tile-batch-size", "4",
  "--score-threshold", $ScoreThreshold,
  "--duplicate-center-distance", "12",
  "--duplicate-overlap-threshold", "0.65",
  "--roi-padding", $RoiPadding
)
if (-not $Overlay) {
  $arguments += "--no-overlay"
}
if ($null -ne $StartSeconds) {
  $arguments += @("--start-seconds", $StartSeconds)
}
if ($null -ne $DurationSeconds) {
  $arguments += @("--duration-seconds", $DurationSeconds)
}

Push-Location $projectRoot
try {
  & $Python @arguments
  if ($LASTEXITCODE -ne 0) {
    throw "ML pipeline exited with code $LASTEXITCODE"
  }
}
finally {
  Pop-Location
}
