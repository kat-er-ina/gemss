# GEMSS Explorer Apps

Interactive [marimo](https://marimo.io/) applications are available for exploring multiple sparse solutions in your data using the GEMSS feature selection algorithm.


## Running the apps

### 1. Install uv
If you do not have `uv` installed, run one of the following commands:

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Launch an app

```bash
# Unlimited version (commercial use allowed)
uv run marimo run app/gemss_explorer_unlimited.py

# Non-commercial version (with TabPFN modeling)
uv run marimo run app/gemss_explorer_noncommercial.py
```

The app will open in your default web browser at `http://localhost:2718`.

## Available apps

### 1. GEMSS Explorer Unlimited
**File:** `gemss_explorer_unlimited.py`

The full-featured GEMSS workflow for unlimited use, including commercial applications:
- Data upload and preprocessing
- Algorithm configuration and execution
- Solution recovery and visualization
- Simple regression validation (L1/L2)

### 2. GEMSS Explorer Non-Commercial
**File:** `gemss_explorer_noncommercial.py`

Includes all features from the unlimited version plus advanced modeling with [TabPFN](https://huggingface.co/Prior-Labs/tabpfn_2_5) for comprehensive solution evaluation.

⚠️ **Important:** Use of TabPFN requires agreement with its [license terms](https://huggingface.co/Prior-Labs/tabpfn_2_5#licensing).

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

## Workflow overview

1. **Configure outputs** — Set save directory and file names
2. **Upload data** — Load your CSV and select index/target columns
3. **Configure algorithm** — Set number of components, sparsity, and optimization parameters
4. **Run feature selection** — Execute Bayesian inference to discover multiple solutions
5. **Assess convergence** — Review ELBO and feature trajectory plots
6. **Recover solutions** — Extract sparse feature sets using different strategies
7. **Evaluate solutions** — Validate with simple regression (and TabPFN if using non-commercial version)

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
