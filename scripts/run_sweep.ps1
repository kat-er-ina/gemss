# PowerShell script for running predefined experiment sweeps (Windows compatible)

# --- Define your combinations below ---
# Each line in $combinations:
#   NSAMPLES, NFEATURES, NSOLUTIONS, SPARSITY, NOISE_STD, N_COMPONENTS
$combinations = @(
    "30,60,3,2,0.01,6"
    "30,60,3,2,0.1,6"
    "40,400,3,4,0.01,6"
    "40,400,3,4,0.01,10"
    "50,600,3,4,0.1,6"
    "50,200,3,5,0.01,6"
    "50,200,3,5,0.01,12"
    "50,200,3,5,0.1,12"
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
$LAMBDA_JACCARD = 100.0
$BATCH_SIZE = 16
$LEARNING_RATE = 0.002
$MIN_MU_THRESHOLD = 0.25
$BINARIZE = $true
$BINARY_RESPONSE_RATIO = 0.5
$RANDOM_SEED = 42

# --- Paths (adjusted for scripts folder location) ---
$SCRIPT_DIR = Split-Path -Parent $MyInvocation.MyCommand.Definition
$ROOT_DIR = Split-Path -Parent $SCRIPT_DIR
$GEN_JSON = Join-Path $ROOT_DIR "generated_dataset_parameters.json"
$ALG_JSON = Join-Path $ROOT_DIR "algorithm_settings.json"
$POST_JSON = Join-Path $ROOT_DIR "solution_postprocessing_settings.json"
$RUN_SCRIPT = Join-Path $SCRIPT_DIR "run_experiment.py"

# --- Ensure results directory exists ---
$resultsDir = Join-Path $ROOT_DIR "results"
if (-not (Test-Path $resultsDir)) {
    New-Item -ItemType Directory -Path $resultsDir | Out-Null
}
Write-Host "Results will be saved in: $resultsDir"

foreach ($combo in $combinations) {
    if ([string]::IsNullOrWhiteSpace($combo)) { continue }
    $parts = $combo.Split(",")
    $NSAMPLES = $parts[0]
    $NFEATURES = $parts[1]
    $NSOLUTIONS = $parts[2]
    $SPARSITY = $parts[3]
    $NOISE_STD = $parts[4]
    $N_COMPONENTS = $parts[5]

    # DESIRED_SPARSITY and PRIOR_SPARSITY always equal SPARSITY
    $DESIRED_SPARSITY = $SPARSITY
    $PRIOR_SPARSITY = $SPARSITY

    # Write generated_dataset_parameters.json
    $genJsonContent = @{
        "NSAMPLES" = [int]$NSAMPLES
        "NFEATURES" = [int]$NFEATURES
        "NSOLUTIONS" = [int]$NSOLUTIONS
        "SPARSITY" = [int]$SPARSITY
        "NOISE_STD" = [double]$NOISE_STD
        "BINARIZE" = [bool]$BINARIZE
        "BINARY_RESPONSE_RATIO" = [double]$BINARY_RESPONSE_RATIO
        "RANDOM_SEED" = [int]$RANDOM_SEED
    } | ConvertTo-Json -Depth 2
    Set-Content -Path $GEN_JSON -Value $genJsonContent

    # Write algorithm_settings.json
    $algJsonContent = @{
        "N_COMPONENTS" = [int]$N_COMPONENTS
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
    Write-Host "NSAMPLES=$NSAMPLES, NFEATURES=$NFEATURES, NSOLUTIONS=$NSOLUTIONS, SPARSITY=$SPARSITY, NOISE_STD=$NOISE_STD, N_COMPONENTS=$N_COMPONENTS"
    Write-Host "====================================================================================="

    # Compose output file name with param settings and timestamp:
    $timestamp = (Get-Date -Format "yyyyMMdd_HHmm")
    $combo_named = @(
        "NSAMPLES=$NSAMPLES"
        "NFEATURES=$NFEATURES"
        "NSOLUTIONS=$NSOLUTIONS"
        "SPARSITY=$SPARSITY"
        "NOISE_STD=$NOISE_STD"
        "N_COMPONENTS=$N_COMPONENTS"
    ) -join "_"
    $output_file = "${resultsDir}\experiment_output_${timestamp}_${combo_named}.txt"

    # Change to root directory before running the experiment
    Push-Location $ROOT_DIR
    try {
        python $RUN_SCRIPT --output $output_file
    } finally {
        Pop-Location
    }
}
Write-Host "====================================================================================="
Write-Host "All predefined experiments finished. Check the results/ directory for outputs."