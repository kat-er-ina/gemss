# List of things to improve and develop further


## Fixes

- **sparsity gap test** - an essential fix: overly pessimistic results, might require major changes.
- **nan_ratio check in regression computations** - nan_ratio is sometimes in range 0 to 100 (instead of 0 - 1) for some components but it is not clear why. It is working, though.
- **fix duplicities in logging** of experiment results.


## Additions

### Postprocessing: analysis of variance

- Add some analysis of explained variance in data by candidate solutions.

### Postprocessing improvements

- Add global ranking of features across all components.
- Possibly test for differences between outliers and other solutions => it can provide some diagnostics.

### Dependencies of parameters (nice to have)

- Some parameters are used only in certain setups, most importantly the type of prior. Currently, all parameters are set up regardless of these dependencies (and only the relevant ones are used). It would be good to have handling of these dependencies in the code instead of just comments.

### Integration & Deployment

- **Marimo notebooks** - interactive web-ready interface. Main purposes: 1. demo, 2. easy access by domain experts
- **Scikit-learn compatibility** - standard fit/predict/transform interface
- **MLflow integration** - experiment tracking and model registry (wishful thinking)

### Better dependency management

- Marek: If you consider pushing the package to PyPi, I would consider moving dependencies that are not needed for running the main code to `dev` dep group. Reasoning, if I want to use `gemss` in some data pipeline I don't want `jupyter` to be installed. `dev` deps are also good place for `ruff`, `pytest`

## Research & Development

- **meta-learning** - learning/automatic setup of optimal hyperparameters

