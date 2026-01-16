# List of things to improve and develop further


## Fixes

- **tabpfn evaluation** uses only the outer_cv parameter but inner_cv is there too, redundant
- **sparsity gap test** - an essential fix: overly pessimistic results, might require major changes.
- **nan_ratio check in regression computations** - nan_ratio is sometimes in range 0 to 100 (instead of 0 - 1) for some components but it is not clear why.
- **fix duplicities in logging** of experiment results.

---

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

---

## Research & Development

- **meta-learning** - learning/automatic setup of optimal hyperparameters

