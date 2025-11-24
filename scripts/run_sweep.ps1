# PowerShell script for running predefined experiment sweeps (Windows compatible)

# --- Define your combinations below ---
# Each line in $combinations:
#   N_SAMPLES, N_FEATURES, N_GENERATING_SOLUTIONS, SPARSITY, NOISE_STD, NAN_RATIO, N_CANDIDATE_SOLUTIONS, LAMBDA_JACCARD
$combinations = @(
    "30,60,3,3,0.1,0.0,6,500, 16, 0.5",      # Small scale, sparsity=3
    "30,60,3,2,0.5,0.0,6,500, 16, 0.5",      # Same but sparsity=2, higher noise
    "40,200,3,3,0.1,0.0,8,500, 16, 0.5",     # More features
    "40,1200,3,3,0.1,0.0,8,500, 16, 0.5",    # High-dimensional
    "50,200,3,5,0.1,0.0,10,500, 16, 0.5",    # Higher sparsity
    "200,200,3,5,0.1,0.0,6,500, 16, 0.5"     # n ≈ p scenario
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
$LEARNING_RATE = 0.002
$MIN_MU_THRESHOLD = 0.2
$BINARIZE = $true
$DATASET_SEED = 42 # Seed for generating the artificial dataset
$SAMPLE_MORE_PRIORS_COEFF = 1.0  # Coefficient for sampling more priors


$USE_MEDIAN_FOR_OUTLIER_DETECTION = $false
$OUTLIER_DEVIATION_THRESHOLDS = @(2.0, 2.5, 3.0)

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

$iteration_counter = 0
foreach ($combo in $combinations) {
    $iteration_counter = $iteration_counter + 1
    if ([string]::IsNullOrWhiteSpace($combo)) { continue }
    $parts = $combo.Split(",")
    $N_SAMPLES = $parts[0]
    $N_FEATURES = $parts[1]
    $N_GENERATING_SOLUTIONS = $parts[2]
    $SPARSITY = $parts[3]
    $NOISE_STD = $parts[4]
    $NAN_RATIO = $parts[5]
    $N_CANDIDATE_SOLUTIONS = $parts[6]
    $LAMBDA_JACCARD = $parts[7]
    $BATCH_SIZE = $parts[8]
    $BINARY_RESPONSE_RATIO = $parts[9]

    # DESIRED_SPARSITY and PRIOR_SPARSITY shall always equal SPARSITY
    $DESIRED_SPARSITY = $SPARSITY
    $PRIOR_SPARSITY = $SPARSITY

    # Write generated_dataset_parameters.json
    $genJsonContent = @{
        "N_SAMPLES" = [int]$N_SAMPLES
        "N_FEATURES" = [int]$N_FEATURES
        "N_GENERATING_SOLUTIONS" = [int]$N_GENERATING_SOLUTIONS
        "SPARSITY" = [int]$SPARSITY
        "NOISE_STD" = [double]$NOISE_STD
        "NAN_RATIO" = [double]$NAN_RATIO
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
        "SAMPLE_MORE_PRIORS_COEFF" = [double]$SAMPLE_MORE_PRIORS_COEFF
    } | ConvertTo-Json -Depth 2
    Set-Content -Path $ALG_JSON -Value $algJsonContent

    # Write solution_postprocessing_settings.json
    $postJsonContent = @{
        "DESIRED_SPARSITY" = [int]$DESIRED_SPARSITY
        "MIN_MU_THRESHOLD" = [double]$MIN_MU_THRESHOLD
        "USE_MEDIAN_FOR_OUTLIER_DETECTION" = [bool]$USE_MEDIAN_FOR_OUTLIER_DETECTION
        "OUTLIER_DEVIATION_THRESHOLDS" = $OUTLIER_DEVIATION_THRESHOLDS
    } | ConvertTo-Json -Depth 2
    Set-Content -Path $POST_JSON -Value $postJsonContent

    Write-Host "\n=====================================================================================\n"
    Write-Host "Running experiment $iteration_counter of $($combinations.Count):"
    Write-Host "N_SAMPLES = $N_SAMPLES, "
    Write-Host "N_FEATURES = $N_FEATURES, "
    Write-Host "N_GENERATING_SOLUTIONS = $N_GENERATING_SOLUTIONS, "
    Write-Host "SPARSITY = $SPARSITY, "
    Write-Host "NOISE_STD = $NOISE_STD, "
    Write-Host "NAN_RATIO = $NAN_RATIO, "
    Write-Host "N_CANDIDATE_SOLUTIONS = $N_CANDIDATE_SOLUTIONS, "
    Write-Host "LAMBDA_JACCARD = $LAMBDA_JACCARD"
    Write-Host "DATASET_SEED = $DATASET_SEED"
    Write-Host "====================================================================================="

    # Compose output file name with param settings and timestamp:
    $timestamp = (Get-Date -Format "yyyyMMdd_HHmm")
    $combo_named = @(
        "N_SAMPLES=$N_SAMPLES"
        "N_FEATURES=$N_FEATURES"
        "N_GENERATING_SOLUTIONS=$N_GENERATING_SOLUTIONS"
        "SPARSITY=$SPARSITY"
        "NOISE_STD=$NOISE_STD"
        "NAN_RATIO=$NAN_RATIO"
        "N_CANDIDATE_SOLUTIONS=$N_CANDIDATE_SOLUTIONS",
        "LAMBDA_JACCARD=$LAMBDA_JACCARD",
        "DATASET_SEED=$DATASET_SEED"
    ) -join "_"
    $output_file = "${resultsDir}\experiment_output_${timestamp}_${combo_named}.txt"

    python $RUN_SCRIPT --output $output_file
}
Write-Host "====================================================================================="
Write-Host "All predefined experiments finished. Check the results/ directory for outputs."