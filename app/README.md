# GEMSS Explorer Apps

Interactive [marimo](https://marimo.io/) applications are available for exploring multiple sparse solutions in your data using the GEMSS feature selection algorithm.


## Running the apps

### 1. Clone this repository

First, you need to download this repository to your computer.

**If you have Git installed:**
```bash
git clone https://github.com/kat-er-ina/gemss.git
cd gemss
```

**If you don't have Git:**
1. Go to the [repository page](https://github.com/kat-er-ina/gemss)
2. Click the green "Code" button
3. Select "Download ZIP"
4. Extract the ZIP file to a location on your computer
5. Open a terminal/command prompt and navigate to the extracted folder


### 2. Install uv
If you do not have `uv` installed, run the following:

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

The full-featured GEMSS workflow:
- Data upload and preprocessing
- Algorithm configuration and execution
- Solution recovery and visualization
- Preliminary validation with simple regression (L1/L2)
- Comprehensive solution evaluation with nested cross-validation using various scikit-learn models

## Using the apps

The marimo framework provides several built-in features:

- **Export tables:** Click the download icon on any displayed table to export as CSV
- **Interactive plots:** Use Plotly controls (hover for details, click-drag to zoom, double-click to reset, camera icon to download as PNG)
- **Adjustable inputs:** All parameters are modified using interactive widgets - no coding required
- **Responsive interface:** The app automatically updates relevant sections when you change inputs or click run buttons
- **Browser-based:** Works entirely in your web browser
- **Session state:** Your progress is maintained as long as the browser tab stays open

For additional marimo features, see the [marimo documentation](https://docs.marimo.io/).

## Data requirements

- **Format:** CSV file
- **Features:** Numeric columns (missing values supported)
- **Structure:** Features in columns, samples in rows
- **Target:** Must include an index column and a target/label column
- **Task types:** Binary classification or regression
- **Stratification (optional):** May include a column for custom cross-validation stratification (e.g., experimental batches, time periods, patient cohorts...)

## Workflow overview

1. **Configure outputs** — Set save directory and file names
2. **Upload data** — Load your CSV and select index/target columns
3. **Configure data** — Choose scaling method and optionally enable custom stratification for cross-validation
4. **Configure algorithm** — Set desired number of components, sparsity, and optimization parameters
5. **Run feature selection** — Execute Bayesian inference to discover multiple solutions
6. **Assess convergence** — Review ELBO and feature trajectory plots
7. **Recover solutions** — Extract sparse feature sets using different strategies
8. **Preliminary solution evaluation** — Quickly validate with simple regression
9. **Predictive modeling metrics** - Assess metrics of various predictive models based on candidate solutions

## Output files

When saving is enabled, results are saved to `experiment_<ID>/` folder at a custom location:
- `search_history_results.json` — Complete optimization history
- `search_setup.json` — Algorithm configuration and constants
- `all_candidate_solutions.json` — Recovered feature sets (JSON)
- `all_candidate_solutions.txt` — Recovered feature sets (human-readable)

## Help

- Each app section includes expandable help panels (📖 icons) to guide a user through hyperparameter setting and interpretation of results.
- See the main [repository README](../README.md) for detailed documentation
- Try the [demo notebook](../notebooks/demo.ipynb) for step-by-step examples
