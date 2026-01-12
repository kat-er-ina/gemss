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

### Integration & Deployment (wishful thinking)

- **Scikit-learn compatibility** - standard fit/predict/transform interface
- **Docker containerization** - reproducible environments
- **MLflow integration** - experiment tracking and model registry

---

## Research & Development

- **meta-learning** - learning/automatic setup of optimal hyperparameters

