# List of things to improve and develop further


## Fixes

- **sparsity gap test** - an essential fix before release: overly pessimistic results, might require major changes

---

## Refactoring

- refactor/reorganize *utils.py*
- fix duplicities in logging of experiment results
- split *diagnostics/* directory into *diagnostics/* and *postprocessing/*
- **type hints completion** - ensure all functions have proper type annotations
- clean up unused parts of code

### Choice of solutions

- Improve on *run_experiment.py* to include and evaluate all 3 types of solutions
- Possibly test for differences between outliers and other solutions => it can provide some diagnostics.


---

## Additions


### Experiment design

- Design the set of parameters, organized in tiers, to be tested in experiments.
- Add the dataset generation random seed as a parameter to log.


### Logging of experiments

- Log basic metrics of every experiment in a single file.
- Add utilities that will process the log file.
- Export diagnostic figures (artifacts) during experiments.


### Guide to adjust lambda of Jaccard penalty

The Jaccard penalty is a value between 0 and 1. Its effect is given by the corresponding parameter *lambda*, which should be set up in accordance with

### Postprocessing improvements

- Real-time monitoring during optimization
- Better visualization and understanding of solution diversity
- To consider: add global ranking of features across all components.

### Integration & Deployment (wishful thinking)

- **Scikit-learn compatibility** - standard fit/predict/transform interface
- **Docker containerization** - reproducible environments
- **MLflow integration** - experiment tracking and model registry

---

## Research & Development

- **meta-learning** - learning/automatic setup optimal hyperparameters

