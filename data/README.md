# Data Directory

This directory is where you should place your CSV datasets for analysis with GEMSS (Gaussian Ensemble for Multiple Sparse Solutions).

## Usage

### For Custom Datasets

1. **Place your CSV file** in this directory
2. **Open** `notebooks/explore_unknown_dataset.ipynb`
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
- **Target column** with the response variable you want to predict
- **Feature columns** with numerical data (categorical features should be encoded)

### Example File Structure

```csv
sample_id,feature_1,feature_2,feature_3,...,target
sample_001,1.23,4.56,7.89,...,0
sample_002,2.34,5.67,8.90,...,1
sample_003,3.45,6.78,9.01,...,0
...
```

## Example Datasets

- Example datasets can be generated directly in the `demo.ipynb` notebook.
- To govern the parameters of dataset generation, edit the `gemss/config/generated_dataset_parameters.json` file
- By default, the example datasets are not being saved

### Data Preprocessing

The `explore_unknown_dataset.ipynb` notebook includes optional data preprocessing:
- **Standard scaling** of features (can be enabled/disabled)
- **Automatic handling** of binary vs continuous targets
- **Feature name mapping** for better interpretability

### Supported Formats

- **File type:** CSV (.csv)
- **Target types:** Binary classification (0/1) or continuous regression
- **Features:** Numerical values (continuous or discrete)
- **Missing values:** Basic `dropna` handling can be done in the notebook, otherwise implement separately

## Tips

- **File size:** GEMSS works well with high-dimensional data (more features than samples)
- **Feature names:** Use descriptive names for better results interpretation
- **Target encoding:** Binary targets should use 0/1 encoding
- **Backup:** Keep a backup of your original data before any preprocessing

## Troubleshooting

**Common issues:**
- **File not found:** Make sure your CSV file is in this `data/` directory
- **Column names:** Ensure `index_column_name` and `label_column_name` match your CSV headers exactly
- **Data types:** Features should be numerical; categorical variables need encoding first

For more details, see the main [README.md](../README.md) and follow the `demo.ipynb` and `explore_unknown_dataset.ipynb` notebooks.
