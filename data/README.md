# Data Directory

This directory is where you should place your CSV datasets for analysis with GEMSS (Gaussian Ensemble for Multiple Sparse Solutions).

## Usage

### For Custom Datasets

1. **Place your CSV file** in this directory
2. **Open** `notebooks/explore_custom_dataset.ipynb`
3. **Configure** the dataset parameters in the notebook:
   ```python
   csv_dataset_name = "your_dataset.csv"
   index_column_name = "sample_id"  # or whatever your index column is called
   label_column_name = "target"     # or whatever your target column is called
   ```
4. **Run** the notebook cells to analyze your data

### File Requirements

Your CSV file should have:
- **Header row** with feature names
- **Index column** with unique sample identifiers (can be any name)
- **Target column** with the response variable you want to predict: either binary (encoded 0/1) or continuous.
- **Feature columns** with numerical data (categorical features should be encoded)

Optionally, the data may contain a column with denoted groups for cross-validation (**stratification groups**). However, such column must be removed before GEMSS feature selector is run. The application handles it natively but the notebooks do *not*.

**Missing values:** The algorithm handles natively handles missing values. It only requires the response vector to be complete, which can be achieved by using a NA-dropping parameter during preprocessing.

### Example File Structure

```csv
sample_id,feature_1,feature_2,feature_3,...,target
sample_001,1.23,4.56,7.89,...,0
sample_002,2.34,5.67,8.90,...,1
sample_003,3.45,6.78,9.01,...,0
...
```

### Data Preprocessing

The `explore_custom_dataset.ipynb` notebook includes optional data preprocessing:
- **Standard/minmax scaling** of features (can be enabled/disabled)
- **Automatic handling** of binary vs continuous targets


## Artificial Datasets

- Artificial example datasets can be generated directly in the `demo.ipynb` notebook.
- To govern the parameters of dataset generation, edit the `gemss/config/generated_dataset_parameters.json` file.
- By default, the example datasets are not being saved.


## Preprocessed Real-World Datasets

The `preprocessed_datasets/` folder contains curated real-world datasets ready for GEMSS analysis. These datasets were preprocessed in the [gemss_benchmarking](https://github.com/kat-er-ina/gemss_benchmarking) repository and are provided here for convenient testing and benchmarking.

### Available Datasets

**Metabolomics Datasets** (from [MetaboLights](https://www.ebi.ac.uk/metabolights/)):

- **MTBLS1 - Type 2 Diabetes** (`diabetes_MTBLS1_preprocessed_n=132_p=222_*.csv`)
  - Binary classification: diabetes vs. healthy
  - 132 samples, ~222 metabolite features
  - NMR-based urinary metabolite profiles
  - Target: `Factor Value[Metabolic syndrome]`
  - **Challenge:** Classic $n << p$ problem

- **MTBLS2 - Arabidopsis Genotype** (`arabidopsis_MTBLS2_preprocessed_n=16_p=41_*.csv`)
  - Binary classification: wild-type vs. knockout genotype
  - 16 samples, 41 identified metabolite features
  - LC/MS-based plant metabolomics
  - Target: `Factor Value[Genotype]`
  - **Challenge:** Extremely low sample size (n=16)

- **MTBLS12968 - PCOS and Preterm Birth** (`pcos_MTBLS12968_preprocessed_n=149_p=488_*.csv`)
  - Two independent binary classification tasks:
      - Polycystic ovary syndrome vs. control
      - Preterm delivery vs. term delivery
  - 149 samples in 4 groups, ~488 metabolite features
  - UHPLC-QTRAP MS/MS plasma profiles
  - Targets: `PCOS` (PCOS vs. control) or `PRETERM` (preterm vs. term delivery)
  - **Multi-task potential:** Two related outcomes from the same samples, each task controls for the other variable
  - **Challenge:** relatively low data signal

### Using Preprocessed Datasets

These datasets are ready to use with both the **GEMSS Explorer app** and the **Jupyter notebooks**:

**With GEMSS Explorer:**
```bash
uv run marimo run app/gemss_explorer.py
```
Upload any CSV file from `data/preprocessed_datasets/` and select the appropriate target column (`response`) and stratification column (`stratification_groups`).

**With Jupyter notebooks:**
Use `explore_custom_dataset.ipynb` and set the `csv_dataset_name` to the desired preprocessed file. Extract the stratification column before running the feature selector.

### Dataset Details

For comprehensive information including:
- Detailed dataset descriptions and characteristics
- Data source citations
- Preprocessing procedures
- Multi-task analysis suggestions

See the [gemss_benchmarking repository documentation](https://github.com/kat-er-ina/gemss_benchmarking/tree/main/real_world_applications).


## Tips

- **File size:** GEMSS works well with high-dimensional data (more features than samples) and small sample sizes (as few as 15 samples).
- **Target encoding:** Binary targets should use 0/1 encoding, otherwise continuous regression is assumed.

For more details, see the main [README.md](../README.md) and follow the `demo.ipynb` and `explore_custom_dataset.ipynb` notebooks.
