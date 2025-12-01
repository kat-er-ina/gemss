# PowerShell script for running predefined experiment sweeps with tier support (Windows compatible)
param(
    [Parameter(Mandatory=$true)]
    [string]$tier,
    [string]$parametersFile = "experiment_parameters.json"
)

# --- Load experiment parameters ---
if (-not (Test-Path $parametersFile)) {
    Write-Error "Parameters file '$parametersFile' not found. Please ensure the file exists in the current directory."
    exit 1
}

Write-Host "Loading experiment parameters from: $parametersFile"
$experimentData = Get-Content $parametersFile -Raw | ConvertFrom-Json

# Validate tier selection
$tierKey = "tier$tier"
if (-not $experimentData.tiers.PSObject.Properties.Name -contains $tierKey) {
    $availableTiers = $experimentData.tiers.PSObject.Properties.Name -join ", "
    Write-Error "Invalid tier '$tier'. Available tiers: $availableTiers"
    exit 1
}

$selectedTier = $experimentData.tiers.$tierKey
$combinations = $selectedTier.combinations
$algorithmParams = $selectedTier.algorithm_parameters
$fixedParams = $experimentData.fixed_parameters

Write-Host "Selected Tier: $tier"
Write-Host "Tier Name: $($selectedTier.name)"
Write-Host "Description: $($selectedTier.description)"
Write-Host "Number of combinations: $($combinations.Count)"
Write-Host "Parameter format: $($experimentData.parameter_format)"
Write-Host "Response type: $(if ($algorithmParams.BINARIZE) { 'Binary Classification' } else { 'Regression' })"
Write-Host ""

# --- Load algorithm parameters from tier configuration ---
$N_ITER = $algorithmParams.N_ITER
$PRIOR_TYPE = $algorithmParams.PRIOR_TYPE
$STUDENT_DF = $algorithmParams.STUDENT_DF
$STUDENT_SCALE = $algorithmParams.STUDENT_SCALE
$VAR_SLAB = $algorithmParams.VAR_SLAB
$VAR_SPIKE = $algorithmParams.VAR_SPIKE
$WEIGHT_SLAB = $algorithmParams.WEIGHT_SLAB
$WEIGHT_SPIKE = $algorithmParams.WEIGHT_SPIKE
$IS_REGULARIZED = $algorithmParams.IS_REGULARIZED
$LEARNING_RATE = $algorithmParams.LEARNING_RATE
$MIN_MU_THRESHOLD = $algorithmParams.MIN_MU_THRESHOLD
$BINARIZE = $algorithmParams.BINARIZE

# --- Load fixed parameters ---
$DATASET_SEED = $fixedParams.DATASET_SEED
$SAMPLE_MORE_PRIORS_COEFF = $fixedParams.SAMPLE_MORE_PRIORS_COEFF
$USE_MEDIAN_FOR_OUTLIER_DETECTION = $fixedParams.USE_MEDIAN_FOR_OUTLIER_DETECTION
$OUTLIER_DEVIATION_THRESHOLDS = $fixedParams.OUTLIER_DEVIATION_THRESHOLDS

Write-Host "Algorithm Parameters for Tier $tier :"
Write-Host " - N_ITER: $N_ITER"
Write-Host " - PRIOR_TYPE: $PRIOR_TYPE" 
Write-Host " - BINARIZE: $BINARIZE"
Write-Host " - LEARNING_RATE: $LEARNING_RATE"
Write-Host " - IS_REGULARIZED: $IS_REGULARIZED"
Write-Host ""

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
$tierResultsDir = Join-Path (Join-Path $currentDir "results") "tier$tier"
if (-not (Test-Path $tierResultsDir)) {
    New-Item -ItemType Directory -Path $tierResultsDir -Force | Out-Null
}
Write-Host "Results will be saved in: $tierResultsDir"

$iteration_counter = 0
$total_experiments = $combinations.Count

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

    # --- PROGRESS BAR UPDATE ---
    $percentComplete = [int](($iteration_counter / $total_experiments) * 100)
    $statusMsg = "Experiment $iteration_counter of $total_experiments"
    $currentOp = "n=$N_SAMPLES, p=$N_FEATURES, k=$SPARSITY, batch=$BATCH_SIZE"
    Write-Progress -Activity "Running Tier $tier Experiments" -Status $statusMsg -PercentComplete $percentComplete -CurrentOperation $currentOp

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

    Write-Host "`n=====================================================================================`n"
    Write-Host "Running experiment $iteration_counter of $($combinations.Count) for Tier $tier :"
    Write-Host "N_SAMPLES = $N_SAMPLES, "
    Write-Host "N_FEATURES = $N_FEATURES, "
    Write-Host "N_GENERATING_SOLUTIONS = $N_GENERATING_SOLUTIONS, "
    Write-Host "SPARSITY = $SPARSITY, "
    Write-Host "NOISE_STD = $NOISE_STD, "
    Write-Host "NAN_RATIO = $NAN_RATIO, "
    Write-Host "N_CANDIDATE_SOLUTIONS = $N_CANDIDATE_SOLUTIONS, "
    Write-Host "LAMBDA_JACCARD = $LAMBDA_JACCARD"
    Write-Host "BATCH_SIZE = $BATCH_SIZE"
    Write-Host "BINARIZE = $BINARIZE"
    Write-Host "DATASET_SEED = $DATASET_SEED"
    Write-Host "====================================================================================="

    # Compose output file name with param settings and timestamp:
    $timestamp = (Get-Date -Format "yyyy-MM-dd-HHmm")
    $responseType = if ($BINARIZE) { "BINARY" } else { "REGRESSION" }
    $combo_named = @(
        "TIER=$tier"
        "N_SAMPLES=$N_SAMPLES"
        "N_FEATURES=$N_FEATURES"
        "N_GENERATING_SOLUTIONS=$N_GENERATING_SOLUTIONS"
        "SPARSITY=$SPARSITY"
        "NOISE_STD=$NOISE_STD"
        "NAN_RATIO=$NAN_RATIO"
        "N_CANDIDATE_SOLUTIONS=$N_CANDIDATE_SOLUTIONS",
        "LAMBDA_JACCARD=$LAMBDA_JACCARD",
        "TYPE=$responseType",
        "DATASET_SEED=$DATASET_SEED"
    ) -join "_"
    $output_file = "${tierResultsDir}\experiment_output_${timestamp}_${combo_named}.txt"

    python $RUN_SCRIPT --output $output_file
}

# Clear the progress bar when done
Write-Progress -Activity "Running Tier $tier Experiments" -Completed

Write-Host "====================================================================================="
Write-Host "All Tier $tier experiments finished. Check the results/tier$tier/ directory for outputs."
Write-Host "Tier summary:"
Write-Host " - Name: $($selectedTier.name)"
Write-Host " - Total experiments: $($combinations.Count)"
Write-Host " - Response type: $(if ($BINARIZE) { 'Binary Classification' } else { 'Regression' })"
Write-Host " - Algorithm: N_ITER=$N_ITER, LR=$LEARNING_RATE"
