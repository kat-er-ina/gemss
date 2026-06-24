# GEMSS Explorer App

Interactive, self-contained [marimo](https://marimo.io/) application is available for exploring multiple sparse solutions in your data using the GEMSS feature selection algorithm. The app also provides downstream modeling for rigorous evaluation of the candidate solutions.

**Do not have data to try the app?** Download one of [preprocessed datasets](https://github.com/kat-er-ina/gemss/tree/main/data) used in our paper.

## Online version

The quickest way to get started is to use the [online version](https://huggingface.co/spaces/kat-er-ina/gemss) hosted at HuggingFace. It does not save the results automatically but the full report of your experiment can be manually exported as HTML.

## Cloned version - quick start

If you need full control of the app, get the full open code by following the steps below.

### 1. Clone this repository

First, you need to download this repository to your computer.

**If you have Git installed:**
```bash
git clone https://github.com/kat-er-ina/gemss.git
cd gemss
```

**If you don't use Git:**
1. Go to the [repository page](https://github.com/kat-er-ina/gemss)
2. Click the green "Code" button
3. Select "Download ZIP"
4. Extract the ZIP file to a location on your computer
5. Open a terminal/command prompt and navigate to the extracted folder


### 2. Install uv
`uv` will install everything the app needs for running. If you do not have it, run the following in your command line:

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 3. Launch the app

Finally, run the app from the root 'gemss' folder using `uv run`:

```bash
uv run marimo run app/gemss_explorer.py
```

The app will open in your default web browser at `http://localhost:2718`.

⚠️ **For Windows users:** If you encounter errors, see the [troubleshooting guide](../README.md#2-set-up-the-environment) in the main README.


## The app

**File:** `gemss_explorer.py`

The GEMSS workflow:
- Data upload and basic preprocessing
- Algorithm configuration, execution and visualization of its run
- Solution recovery & preliminary validation
- Comprehensive evaluation of predictive performance of a chosen solution type

## Using the app

The framework provides several built-in features:

- **Browser-based:** Works entirely in your web browser
- **Adjustable inputs:** All parameters are modified using interactive widgets - no coding required
- **In-app guides:** Collapsible help panels to guide you through the setup and interpretaion of results
- **Export report:** Save the full report as HTML for later analysis.
- **Interactivity:** Control displayed plots, export tables..

## Data requirements

The data uploaded to GEMSS explorer must have the standard format as a table with features and response as columns and samples in rows.

- **Format:** CSV file
- **Features:** Numeric columns only (missing values supported)
- **Structure:** Features in columns, samples in rows
- **Target:** Must include an index column and a target/label column
- **Task types:** Binary classification or regression
- **Stratification (optional):** May include a column for custom cross-validation stratification (e.g., experimental batches, time periods, patient cohorts...)


## Output files

Results are automatically saved to `experiment_<ID>/` folder at your specified location.

### Feature selection outputs

Generated after running the feature selection algorithm:

- `search_history_results.json` — Complete optimization history (all parameter trajectories)
- `search_setup.json` — Algorithm configuration and dataset constants
- `all_candidate_solutions.json` — Recovered feature sets in JSON format (for possible further machine processing)
- `all_candidate_solutions.txt` — Recovered feature sets (human-readable text)

### Modeling outputs

Generated after running cross-validated evaluation:

- `model_comparison_solutiontype=<type>.csv` — Performance metrics across all models and components
- `performance_<component_name>_solutiontype=<type>.csv` — Detailed per-model metrics for each component
- `all_models_solutiontype=<type>.txt` — Comprehensive summary of all modeling results

**Note:** `<type>` refers to the solution recovery strategy (e.g., `outliers_2.0`, `outliers_2.5`, `top`), and `<component_name>` is the specific component identifier (e.g., `component_0`, `component_1`).

### Saving the complete report

Since the algorithm is stochastic, it is recommended to save the entire interactive report for reproducibility. Use the **three-dot menu** (...) in the upper right corner of the app to export as HTML.

## Help

- Each app section includes expandable help panels (📖 icons) to guide a user through hyperparameter setting and interpretation of results.
- See the main [repository README](../README.md) for detailed documentation
- Try the [demo notebook](../notebooks/demo.ipynb) for step-by-step examples
