# PowerShell script for running predefined experiment sweeps (Windows compatible)

# --- Define your combinations below ---
# Each line in $combinations:
#   N_SAMPLES, N_FEATURES, N_GENERATING_SOLUTIONS, SPARSITY, NOISE_STD, N_CANDIDATE_SOLUTIONS, LAMBDA_JACCARD
$combinations = @(
    "30,60,3,3,0.1,6,100"
    "30,60,3,3,0.1,6,500"
    "30,60,3,3,0.1,6,1000"
    # "40,200,3,3,0.1,8,500"
    # "40,200,3,3,0.1,8,1000"
    # "40,200,3,3,1.0,8,500"
    # "40,200,3,3,1.0,8,1000"
    # "40,400,3,3,0.1,8,500"
    # "40,600,3,3,0.1,8,500"
    # "40,800,3,3,0.1,8,500"
    # "40,800,3,3,0.1,12,500"
    # "30,60,3,2,0.5,6,500"
    # "30,60,3,2,0.01,6,500"
    # "30,60,3,2,0.1,6,500"
    # "40,400,3,4,0.01,6,500"
    # "40,400,3,4,0.01,10,500"
    # "50,200,3,5,0.01,6,500"
    # "50,200,3,5,0.01,12,500"
    # "50,200,3,5,0.1,12,500"
)

# --- Fixed parameters for the algorithm ---
$N_ITER = 4000
$PRIOR_TYPE = "sss"
$STUDENT_DF = 1
$STUDENT_SCALE = 1.0
$VAR_SLAB = 100.0
$VAR_SPIKE = 0.1
$WEIGHT_SLAB = 0.9
$WEIGHT_SPIKE = 0.1
$IS_REGULARIZED = $true
$BATCH_SIZE = 16
$LEARNING_RATE = 0.002
$MIN_MU_THRESHOLD = 0.2
$BINARIZE = $true
$BINARY_RESPONSE_RATIO = 0.5

# Seed for generating the artificial dataset
$DATASET_SEED = 42

# --- Get paths from Python constants ---
$pythonCmd = @"
import sys
import os
sys.path.insert(0, '..')
from gemss.config.constants import CONFIG_FILES
from pathlib import Path
# Get the parent directory of the current working directory (scripts -> root)
config_dir = Path(os.getcwd()).parent / 'gemss' / 'config'
str_data = 'ARTIFICIAL_DATASET'
str_algo = 'ALGORITHM'
str_post = 'POSTPROCESSING'
print(config_dir / CONFIG_FILES[str_data])
print(config_dir / CONFIG_FILES[str_algo])
print(config_dir / CONFIG_FILES[str_post])
"@
$paths = python -c $pythonCmd
$pathArray = $paths.Split("`n")
$GEN_JSON = $pathArray[0].Trim()
$ALG_JSON = $pathArray[1].Trim()
$POST_JSON = $pathArray[2].Trim()
$RUN_SCRIPT = "run_experiment.py"

# --- Ensure results directory exists ---
$currentDir = Get-Location
$resultsDir = Join-Path $currentDir "results"
if (-not (Test-Path $resultsDir)) {
    New-Item -ItemType Directory -Path $resultsDir | Out-Null
}
Write-Host "Results will be saved in: $resultsDir"

foreach ($combo in $combinations) {
    if ([string]::IsNullOrWhiteSpace($combo)) { continue }
    $parts = $combo.Split(",")
    $N_SAMPLES = $parts[0]
    $N_FEATURES = $parts[1]
    $N_GENERATING_SOLUTIONS = $parts[2]
    $SPARSITY = $parts[3]
    $NOISE_STD = $parts[4]
    $N_CANDIDATE_SOLUTIONS = $parts[5]
    $LAMBDA_JACCARD = $parts[6]

    # DESIRED_SPARSITY and PRIOR_SPARSITY always equal SPARSITY
    $DESIRED_SPARSITY = $SPARSITY
    $PRIOR_SPARSITY = $SPARSITY

    # Write generated_dataset_parameters.json
    $genJsonContent = @{
        "N_SAMPLES" = [int]$N_SAMPLES
        "N_FEATURES" = [int]$N_FEATURES
        "N_GENERATING_SOLUTIONS" = [int]$N_GENERATING_SOLUTIONS
        "SPARSITY" = [int]$SPARSITY
        "NOISE_STD" = [double]$NOISE_STD
        "BINARIZE" = [bool]$BINARIZE
        "BINARY_RESPONSE_RATIO" = [double]$BINARY_RESPONSE_RATIO
        "DATASET_SEED" = [int]$DATASET_SEED
    } | ConvertTo-Json -Depth 2
    Set-Content -Path $GEN_JSON -Value $genJsonContent

    # Write algorithm_settings.json
    $algJsonContent = @{
        "N_CANDIDATE_SOLUTIONS" = [int]$N_CANDIDATE_SOLUTIONS
        "N_ITER" = [int]$N_ITER
        "PRIOR_TYPE" = $PRIOR_TYPE
        "STUDENT_DF" = [int]$STUDENT_DF
        "STUDENT_SCALE" = [double]$STUDENT_SCALE
        "VAR_SLAB" = [double]$VAR_SLAB
        "VAR_SPIKE" = [double]$VAR_SPIKE
        "WEIGHT_SLAB" = [double]$WEIGHT_SLAB
        "WEIGHT_SPIKE" = [double]$WEIGHT_SPIKE
        "IS_REGULARIZED" = [bool]$IS_REGULARIZED
        "LAMBDA_JACCARD" = [double]$LAMBDA_JACCARD
        "BATCH_SIZE" = [int]$BATCH_SIZE
        "LEARNING_RATE" = [double]$LEARNING_RATE
        "PRIOR_SPARSITY" = [int]$PRIOR_SPARSITY
    } | ConvertTo-Json -Depth 2
    Set-Content -Path $ALG_JSON -Value $algJsonContent

    # Write solution_postprocessing_settings.json
    $postJsonContent = @{
        "DESIRED_SPARSITY" = [int]$DESIRED_SPARSITY
        "MIN_MU_THRESHOLD" = [double]$MIN_MU_THRESHOLD
    } | ConvertTo-Json -Depth 2
    Set-Content -Path $POST_JSON -Value $postJsonContent

    Write-Host "====================================================================================="
    Write-Host "Running experiment with:"
    Write-Host "N_SAMPLES = $N_SAMPLES, "
    Write-Host "N_FEATURES = $N_FEATURES, "
    Write-Host "N_GENERATING_SOLUTIONS = $N_GENERATING_SOLUTIONS, "
    Write-Host "SPARSITY = $SPARSITY, "
    Write-Host "NOISE_STD = $NOISE_STD, "
    Write-Host "N_CANDIDATE_SOLUTIONS = $N_CANDIDATE_SOLUTIONS, "
    Write-Host "LAMBDA_JACCARD = $LAMBDA_JACCARD"
    Write-Host "====================================================================================="

    # Compose output file name with param settings and timestamp:
    $timestamp = (Get-Date -Format "yyyyMMdd_HHmm")
    $combo_named = @(
        "N_SAMPLES=$N_SAMPLES"
        "N_FEATURES=$N_FEATURES"
        "N_GENERATING_SOLUTIONS=$N_GENERATING_SOLUTIONS"
        "SPARSITY=$SPARSITY"
        "NOISE_STD=$NOISE_STD"
        "N_CANDIDATE_SOLUTIONS=$N_CANDIDATE_SOLUTIONS",
        "LAMBDA_JACCARD=$LAMBDA_JACCARD"
    ) -join "_"
    $output_file = "${resultsDir}\experiment_output_${timestamp}_${combo_named}.txt"

    python $RUN_SCRIPT --output $output_file
}
Write-Host "====================================================================================="
Write-Host "All predefined experiments finished. Check the results/ directory for outputs."