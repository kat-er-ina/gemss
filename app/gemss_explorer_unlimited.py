import marimo

__generated_with = '0.19.8'
app = marimo.App(width='full')


@app.cell
def _():
    import sys
    import os

    # Add the parent directory to sys.path so 'gemss' can be imported
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
    return current_dir, os


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import io
    from plotly import io as pio

    # GEMSS feature selector
    from gemss.feature_selection.inference import BayesianFeatureSelector

    # Visualizations
    from gemss.utils.visualizations import (
        show_label_histogram,
        show_final_alphas,
        show_features_in_components,
        get_algorithm_progress_plots,
        get_final_alphas_plot,
        get_label_histogram_plot,
        get_label_piechart,
        get_features_in_components_plot,
    )

    # Postprocessing
    from gemss.utils.utils import (
        get_solution_summary_df,
        save_feature_lists_json,
        save_feature_lists_txt,
        save_selector_history_json,
        save_constants_json,
    )
    from gemss.postprocessing.result_postprocessing import (
        recover_solutions,
        get_features_from_solutions,
        get_unique_features,
    )
    from gemss.postprocessing.simple_regressions import (
        detect_task,
        solve_any_regression,
        show_regression_metrics,
    )
    from gemss.postprocessing.result_modeling import (
        evaluate_all_solutions,
        _get_available_models,
    )
    from gemss.data_handling.data_processing import (
        preprocess_features,
        get_df_from_X,
    )

    # Use default renderer or 'iframe' if plots don't show
    # pio.renderers.default = "notebook_connected"
    return (
        BayesianFeatureSelector,
        detect_task,
        evaluate_all_solutions,
        get_algorithm_progress_plots,
        get_df_from_X,
        get_features_from_solutions,
        get_features_in_components_plot,
        get_final_alphas_plot,
        get_label_histogram_plot,
        get_label_piechart,
        get_solution_summary_df,
        get_unique_features,
        io,
        mo,
        np,
        pd,
        preprocess_features,
        recover_solutions,
        save_constants_json,
        save_feature_lists_json,
        save_feature_lists_txt,
        save_selector_history_json,
        solve_any_regression,
    )


@app.cell
def _(current_dir, mo, os):
    logo_path = os.path.join(current_dir, 'datamole_logo_wide.jpg')

    # Read and display logo
    logo_link = mo.Html(
        f"""
        <div style="text-align: right;">
            <a href="https://www.datamole.ai/" target="_blank">
                {mo.image(src=logo_path, width=1000, alt='Datamole').text}
            </a>
        </div>
    """
    )

    mo.image(src=logo_path, width=1000, alt='Datamole')
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    # 💎 **GEMSS Explorer**

    This app helps you discover **multiple distinct feature sets** that explain your data using GEMSS: Gaussian Ensemble for Multiple Sparse Solutions.
    """
    )
    return


@app.cell
def _(mo):
    # More details
    intro_help = mo.accordion(
        {
            ' 📖 Read more': mo.md(
                """
                ### What you will get

                Instead of finding just one "best" set of features, GEMSS discovers **several most likely feature combinations** that predict your target variable comparably well. This is valuable when:

                - You have precious few samples and many more features (common e.g. in life sciences).
                - Multiple underlying mechanisms might explain your data.
                - You are striving for an interpretable model.
                - You want to engineer a multitude of nonlinear and combined features from your original set for exploratory purposes.
                - Your features are correlated.
                - When there is domain knowledge to be mined (a human in the loop).

                **Example:** Instead of "use features A, B, C", you might discover three solutions: {A, B, C}, {D, E, F}, {A, E, G} — each explaining the data through a different mechanism.

                ### General workflow overview

                **I. Data loading** - Upload and configure your dataset. <br>
                **II. Algorithm setup** - Configure hyperparameters of GEMSS feature selector. <br>
                **III. Feature selection** - Run Bayesian inference to discover multiple components that describe your data. <br>
                **IV. Solution recovery** - Extract one sparse solution from each component, obtaining an ensemble of feature sets. <br>
                **V. Preliminary evaluation** - Validate each solution by a quick linear/logistic regression model. <br>
                **VI. Full predictive modeling** - Evaluate the chosen solution type by training and testing full predictive models (e.g. random forest, gradient boosting) with nested cross-validation. <br>

                Each step builds on the previous one, so please follow the workflow in order.
                """
            )
        }
    )

    mo.vstack(
        [
            intro_help,
            mo.md('<br>'),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## **1. Set up input and output**
    """
    )
    return


@app.cell
def _(mo):
    # More details
    data_help = mo.accordion(
        {
            ' 📖 Guide': mo.md(
                """
                ### The input data
                - must be already cleaned and preprocessed,
                - should have features in columns and samples in rows,
                - must contain an index column,
                - must contain a target/label column: binary classification and regression are supported,
                - can contain missing values,
                - only numeric features are supported,
                - may contain a column used for stratification during cross-validation.
                
                ### Stratification in cross-validation
                
                Stratification ensures that each CV fold maintains the same distribution as the full dataset.
                
                **Default behavior:**
                - Classification tasks: stratified by target/label column (preserves class distribution)
                - Regression tasks: no stratification (random splits)
                
                **Custom stratification column:**
                - Enable the "Use custom stratification column" checkbox to specify a different column for stratification
                - Useful when samples have inherent grouping (experimental batches, time periods, patient cohorts, etc.)
                - The stratification column values should be categorical or discrete
                - If disabled, default behavior is used

                ### Data scaling

                It is highly recommended to scale your data, unless they already have been.
                - Use the *minmax* scaling to simply squish all features' values to [0, 1] range.
                - Use the *standard* scaling when you assume the samples are normally distributed. Each feature will then have mean 0 and standard deviation 1.

                ### The outputs

                When saving is enabled, the following files will be created in the specified directory (names can be customized):
                - `search_history_results.json` - JSON file containing the history of the feature selection process (backup for possible alternative postprocessing)
                - `search_setup.json` - JSON file containing the configuration of the feature selection process (for experiment tracking)
                - `feature_lists.txt` - human-readable text file containing the discovered feature sets
                - `feature_lists.json` - same as above, but in JSON format for easier programmatical handling in further modeling
                """
            )
        }
    )

    mo.vstack(
        [
            data_help,
        ]
    )
    return


@app.cell
def _(current_dir, mo):
    # Save configuration
    checkbox_save_results = mo.ui.checkbox(
        value=True,
        label='Save results',
        disabled=True,
    )
    save_dir_input = mo.ui.text(
        value=f'{current_dir}\\results',
        label='Parent directory for saving this experiment',
    )

    save_experiment_id = mo.ui.number(1, 1000, value=1, step=1, label='Experiment ID')
    save_history_name = mo.ui.text(
        value='search_history_results',
        label='History filename (no extension)',
    )
    save_setup_name = mo.ui.text(value='search_setup', label='Setup filename (no extension)')
    save_features_name = mo.ui.text(
        value='all_candidate_solutions', label='Features filename (no extension)'
    )

    mo.vstack(
        [
            mo.md('### 1.1 Output configuration'),
            checkbox_save_results,
        ]
    )
    return (
        checkbox_save_results,
        save_dir_input,
        save_experiment_id,
        save_features_name,
        save_history_name,
        save_setup_name,
    )


@app.cell
def _(
    checkbox_save_results,
    mo,
    save_dir_input,
    save_experiment_id,
    save_features_name,
    save_history_name,
    save_setup_name,
):
    # configure saving options, if saving is enabled
    save_results = checkbox_save_results.value
    if save_results:
        _display = mo.vstack(
            [
                mo.md(
                    f'*Output files will be saved in: {save_dir_input.value}/experiment_{save_experiment_id.value}*/'
                ),
                mo.accordion(
                    {
                        ' Edit save location': mo.vstack(
                            [
                                save_dir_input,
                                save_experiment_id,
                            ]
                        )
                    }
                ),
                mo.accordion(
                    {
                        'Edit file names': mo.vstack(
                            [
                                save_history_name,
                                save_setup_name,
                                save_features_name,
                            ]
                        )
                    }
                ),
            ]
        )
    else:
        _display = None

    _display
    return (save_results,)


@app.cell
def _(mo):
    # UI Components for Data Loading
    file_uploader = mo.ui.file(kind='button', label='Upload CSV dataset', filetypes=['.csv'])

    mo.vstack(
        [
            mo.md('### 1.2 Input data'),
            file_uploader,
        ]
    )
    return (file_uploader,)


@app.cell
def _(file_uploader, io, mo, pd):
    # Logic to load data from uploader
    if file_uploader.value:
        # Load CSV from the uploaded bytes
        _content = file_uploader.value[0].contents
        df_raw = pd.read_csv(io.BytesIO(_content))

        # Column selectors based on the uploaded file
        index_col_selector = mo.ui.dropdown(
            options=list(df_raw.columns),
            label='Index column',
            value=df_raw.columns[0] if not df_raw.empty else None,
        )
        label_col_selector = mo.ui.dropdown(
            options=list(df_raw.columns),
            label='Target/label column',
            value=df_raw.columns[-1] if not df_raw.empty else None,
        )

        # Checkbox to enable custom stratification
        use_custom_stratification = mo.ui.checkbox(
            value=False,
            label='Use custom stratification column',
        )

        stratification_col_selector = mo.ui.dropdown(
            options=list(df_raw.columns),
            label='    → Custom stratification column:',
            value=df_raw.columns[-1] if not df_raw.empty else None,
        )
        scaling_selector = mo.ui.dropdown(
            options=['standard', 'minmax', None],
            label='Scaling to use',
            value='standard',
        )
        allowed_missing_percentage_selector = mo.ui.number(
            0,
            50,
            value=10,
            step=5,
            label='Missing data allowed in a feature [%]',
        )

        data_setup_ui = mo.vstack(
            [
                mo.md('<br>'),
                mo.md(
                    f'✅ **Data loaded:** `{file_uploader.value[0].name}` ({df_raw.shape[0]} rows, {df_raw.shape[1]} cols)'
                ),
                mo.vstack(
                    [
                        index_col_selector,
                        label_col_selector,
                        mo.md('---'),
                        scaling_selector,
                        mo.md('---'),
                        use_custom_stratification,
                        stratification_col_selector,
                    ]
                ),
            ]
        )
    else:
        df_raw = None
        index_col_selector = None
        label_col_selector = None
        use_custom_stratification = None
        stratification_col_selector = None
        scaling_selector = None
        data_setup_ui = None

    (
        mo.vstack(
            [
                mo.md('**Your loaded dataset:**'),
                mo.ui.table(df_raw),
                mo.md('<br>'),
                data_setup_ui,
            ]
        )
        if df_raw is not None
        else mo.vstack(
            [
                mo.md('---'),
            ]
        )
    )
    return (
        allowed_missing_percentage_selector,
        df_raw,
        index_col_selector,
        label_col_selector,
        use_custom_stratification,
        stratification_col_selector,
        scaling_selector,
    )


@app.cell
def _(
    allowed_missing_percentage_selector,
    df_raw,
    get_df_from_X,
    get_label_histogram_plot,
    get_label_piechart,
    index_col_selector,
    label_col_selector,
    mo,
    np,
    pd,
    preprocess_features,
    scaling_selector,
    use_custom_stratification,
    stratification_col_selector,
):
    # Stop if data not loaded
    mo.stop(df_raw is None, mo.md('*Please upload your dataset to proceed.*<br><hr>'))

    # Handle stratification column
    # If custom stratification is enabled and the stratification column is different from label column, extract it
    # Otherwise use None to trigger default backend behavior
    if (
        use_custom_stratification.value
        and stratification_col_selector.value != label_col_selector.value
    ):
        # Extract stratification column WITHOUT modifying df_raw
        stratify_col = df_raw[stratification_col_selector.value].copy()
        cols_to_exclude = [stratification_col_selector.value]
    else:
        stratify_col = None
        cols_to_exclude = []

    # Data Preprocessing
    try:
        _df_proc = df_raw.copy()

        # Drop stratification column if it's separate from label
        for col in cols_to_exclude:
            if col in _df_proc.columns:
                _df_proc = _df_proc.drop(columns=[col])

        if index_col_selector.value:
            _df_proc.set_index(index_col_selector.value, inplace=True)

        _response = _df_proc[label_col_selector.value]
        # Preprocess
        X, y, feature_map = preprocess_features(
            _df_proc,
            _response,
            dropna='response',
            allowed_missing_percentage=allowed_missing_percentage_selector.value,
            drop_non_numeric_features=True,
            apply_scaling=scaling_selector.value,
            verbose=False,
        )
        overall_nan_ratio = np.isnan(X).sum() / (X.shape[0] * X.shape[1])
        df_processed = get_df_from_X(X, feature_map)

        # Filter stratification column to match preprocessing (dropna on response)
        if stratify_col is not None:
            # preprocess_features dropped rows where response is NA
            # Apply same filter to stratification column
            response_values = df_raw[label_col_selector.value]
            valid_mask = response_values.notna()
            stratify_col = stratify_col[valid_mask].reset_index(drop=True)

    except Exception as e:
        mo.stop(True, mo.md(f'**Error processing data:** {str(e)}'))

    n_samples = df_processed.shape[0]
    n_features = df_processed.shape[1]
    _n_response_values = pd.Series(y).nunique()

    min_samples_allowed = 10
    min_features_allowed = 10
    too_few_samples = n_samples < min_samples_allowed
    too_few_features = n_features < min_features_allowed

    mo.stop(
        too_few_samples,
        mo.md(
            f'**Error: too few samples available:** {n_samples} < {min_samples_allowed}. Fix your dataset before proceeding.'
        ),
    )
    mo.stop(
        too_few_features,
        mo.md(
            f'**Error: too few features available:** {n_features} < {min_features_allowed}. Fix your dataset before proceeding.'
        ),
    )

    # Determine stratification description for display
    if stratify_col is not None:
        _n_strat_groups = pd.Series(stratify_col).nunique()
        _strat_desc = f'custom stratification: by column *{stratification_col_selector.value}* ({_n_strat_groups} unique groups)'
    else:
        _task_type = 'classification' if _n_response_values < 10 else 'regression'
        if _task_type == 'classification':
            _strat_desc = f'default stratification: by target column *{label_col_selector.value}* (preserves class distribution)'
        else:
            _strat_desc = (
                'default stratification: none (fully random splits, suitable for regression tasks)'
            )

    mo.vstack(
        [
            mo.md('<br>'),
            mo.md(
                f"""
                ✅ **Data preprocessed:**
                - no. samples: {n_samples}
                - no. features: {n_features}
                - no. unique response values: {_n_response_values}
                - missing data: {overall_nan_ratio}%
                - scaling applied: {scaling_selector.value}
                - {_strat_desc}
                """,
            ),
            # show label distribution either as a pie chart or a histogram, depending on the number of unique values
            (get_label_piechart(y) if _n_response_values < 5 else get_label_histogram_plot(y)),
            mo.md('---'),
            mo.md('<br>'),
        ]
    )
    return X, df_processed, feature_map, n_features, y


@app.cell
def _(mo):
    mo.md(
        r"""
    ## **2. The feature selection algorithm**

    Configure parameters of the GEMSS feature selection algorithm.
    """
    )
    return


@app.cell
def _(df_processed, mo):
    # UI for Algorithm Configuration
    mo.stop(df_processed is None, mo.md('*Please upload data first.*'))

    # Basic Settings
    n_candidates = mo.ui.number(
        1, 20, value=8, step=1, label='Number of components (candidate solutions)'
    )
    sparsity_est = mo.ui.number(
        1,
        50,
        value=4,
        step=1,
        label='Desired sparsity (no. features per component)',
    )

    # Advanced Settings
    adv_iter = mo.ui.number(200, 20000, value=3000, step=100, label='Iterations')
    adv_lr = mo.ui.number(0.0000, 0.1, value=0.002, step=0.0001, label='Learning rate')
    adv_batch = mo.ui.number(
        8,
        256,
        value=16,
        step=4,
        label='Batch size (no. samples in a minibatch)',
    )
    adv_jaccard = mo.ui.checkbox(
        value=True, label='Enforce diversity (penalize Jaccard similarity)'
    )
    adv_lambda = mo.ui.number(0, 20000, value=1000, step=250, label='Lambda')
    adv_var_spike = mo.ui.number(
        0,
        10,
        value=0.1,
        step=0.005,
        label='Spike distribution variance',
    )
    adv_var_slab = mo.ui.number(
        0,
        1000,
        value=100,
        step=10,
        label='Slab distribution variance',
    )

    # Parameter help text
    parameter_help = mo.accordion(
        {
            '📖 Parameter guide': mo.md(
                """
                ### Basic parameters

                - **Number of components:** How many distinct feature sets to discover. It is recommended to overshoot this number (2-3x), especially in adverse conditions.
                - **Estimated sparsity:** Expected number of features per solution. This guides the algorithm's search.

                ### Advanced optimization settings

                - **Iterations:** More iterations improve convergence but take longer (typical: 3000-5000). Increase if ELBO hasn't converged or features' mu values are still changing.
                - **Learning rate:** Controls [SGD optimization](https://en.wikipedia.org/wiki/Stochastic_gradient_descent) step size (typical: 0.001-0.003). Decrease if training is unstable, increase if progress is too slow.
                - **Batch size:** Number of samples used in one SGD optimization step (typical: 16-64). Increase for datasets with missing data, noise, or class imbalance. Increasing batch size proportionally increases run time. Recommendation: have at least 4 samples of the minority class in one batch.
                - **Enforce diversity:** Penalizes average similarity ([Jaccard index](https://en.wikipedia.org/wiki/Jaccard_index)) of solutions to promote diverse feature combinations. Enable when you want to push more towards distinct explanatory mechanisms.
                - **Lambda:** Strength of diversity penalty (typical: 0-2000). Higher values → more different solutions. Increase if solutions overlap too much.

                <i>**Example batch size setup.**
                Let your dataset contain 200 samples: 160x class A, 40x class B (i.e. the minority class makes up 20% samples). There is low noise and no missing data.
                It is desirable to have *at least* 4 samples from each class in a batch (empirical observation) => batch size = 4 * (1/0.20) = 20 samples.
                </i>

                ### Advanced prior settings

                These control how the algorithm balances sparsity (few features) vs. explanatory power (including relevant features).

                This algorithm uses the [Structured Spike-and-Slab](https://en.wikipedia.org/wiki/Spike-and-slab_regression) [prior distribution](https://en.wikipedia.org/wiki/Prior_probability),
                that is a mixture of two Gaussian distributions.
                Each feature is assigned to either the wide distribution (Slab) or the steep distribution (Spike).

                - **Spike variance:**
                    - Is the most important parameter for controlling convergence.
                    - Controls sparsity strength (typical: 0.05-0.5).
                    - Increase if all features converge to 0 (over-regularization).
                    - Decrease carefully if too many features are selected (under-regularization).

                - **Slab variance:**
                    - Scale for non-zero features (typical: 50-200).
                    - Adjust together with spike variance to improve feature discrimination.
                """
            )
        }
    )

    settings_ui = mo.vstack(
        [
            parameter_help,
            mo.md('<br>'),
            mo.vstack([n_candidates, sparsity_est]),
            mo.accordion(
                {
                    'Advanced optimization settings': mo.vstack(
                        [adv_iter, adv_lr, adv_batch, adv_jaccard, adv_lambda]
                    )
                }
            ),
            mo.accordion(
                {
                    'Advanced prior settings (Structured Spike-and-Slab)': mo.vstack(
                        [adv_var_spike, adv_var_slab]
                    )
                }
            ),
        ]
    )
    settings_ui
    return (
        adv_batch,
        adv_iter,
        adv_jaccard,
        adv_lambda,
        adv_lr,
        adv_var_slab,
        adv_var_spike,
        n_candidates,
        sparsity_est,
    )


@app.cell
def _(df_processed, mo):
    mo.stop(df_processed is None, '')

    # The big RUN FEATURE SELECTION button
    run_btn = mo.ui.run_button(label='Run feature selection', kind='success')

    mo.vstack(
        [
            run_btn,
            mo.md('<br>'),
            mo.md('---'),
        ]
    )
    return (run_btn,)


@app.cell
def _(
    BayesianFeatureSelector,
    X,
    adv_batch,
    adv_iter,
    adv_jaccard,
    adv_lambda,
    adv_lr,
    adv_var_slab,
    adv_var_spike,
    df_raw,
    mo,
    n_candidates,
    run_btn,
    sparsity_est,
    y,
):
    # Main execution logic

    # Initialize history to None (in case cell stops early)
    history = None

    # 1. Stop if data not loaded
    mo.stop(df_raw is None, mo.md('*Please upload data first.*'))

    # 2. Stop if button not pressed
    mo.stop(not run_btn.value, mo.md('*Ready to run. Click start above.*'))

    # Optimization: class setup
    selector = BayesianFeatureSelector(
        n_features=X.shape[1],
        n_components=n_candidates.value,
        X=X,
        y=y,
        prior='sss',
        sss_sparsity=sparsity_est.value,
        var_slab=adv_var_slab.value,
        var_spike=adv_var_spike.value,
        lr=adv_lr.value,
        batch_size=adv_batch.value,
        n_iter=adv_iter.value,
    )

    # Run optimizer
    history = selector.optimize(
        regularize=adv_jaccard.value,
        lambda_jaccard=adv_lambda.value,
        verbose=True,
    )
    return (history,)


@app.cell
def _(
    X,
    adv_batch,
    adv_iter,
    adv_jaccard,
    adv_lambda,
    adv_lr,
    adv_var_slab,
    adv_var_spike,
    n_candidates,
    os,
    save_dir_input,
    save_experiment_id,
    save_features_name,
    save_history_name,
    save_results,
    save_setup_name,
    sparsity_est,
):
    # Saving setup
    if save_results:
        # Configure saving options, if saving is enabled
        # Prepare directory
        experiment_dir = f'{save_dir_input.value}/experiment_{save_experiment_id.value}'
        os.makedirs(experiment_dir, exist_ok=True)

        # Prepare save paths
        history_path = f'{experiment_dir}/{save_history_name.value}.json'
        setup_path = f'{experiment_dir}/{save_setup_name.value}.json'
        features_path_json = f'{experiment_dir}/{save_features_name.value}.json'
        features_path_txt = f'{experiment_dir}/{save_features_name.value}.txt'

        # Define constants that are to be saved
        constants = {
            'N_SAMPLES': X.shape[0],
            'N_FEATURES': X.shape[1],
            'N_CANDIDATE_SOLUTIONS': n_candidates.value,
            'SPARSITY': sparsity_est.value,
            'PRIOR_SPARSITY': sparsity_est.value,
            'PRIOR_TYPE': 'sss',
            'VAR_SPIKE': adv_var_spike.value,
            'VAR_SLAB': adv_var_slab.value,
            'N_ITER': adv_iter.value,
            'LEARNING_RATE': adv_lr.value,
            'BATCH_SIZE': adv_batch.value,
            'IS_REGULARIZED': adv_jaccard.value,
            'LAMBDA_JACCARD': adv_lambda.value,
        }
    else:
        experiment_dir = None
        history_path = None
        setup_path = None
        features_path_json = None
        features_path_txt = None
        constants = None
    return (
        constants,
        experiment_dir,
        features_path_json,
        features_path_txt,
        history_path,
        setup_path,
    )


@app.cell
def _(
    constants,
    experiment_dir,
    history,
    history_path,
    mo,
    save_constants_json,
    save_results,
    save_selector_history_json,
    setup_path,
):
    # Save history and setup immediately after optimization
    if save_results:
        # Save history and constants
        msg_history = save_selector_history_json(history, history_path)
        msg_constants = save_constants_json(constants, setup_path)

        # Stack output messages
        _display = mo.vstack(
            [
                mo.md('✅ **Optimization Complete!**'),
                mo.md(f'📁 Optimization history saved to: `{experiment_dir}`'),
                mo.md(f'- {msg_history}'),
                mo.md(f'- {msg_constants}'),
                mo.md('<br>'),
                mo.md('---'),
                mo.md('<br>'),
            ]
        )
    else:
        _display = mo.vstack(
            [
                mo.md('✅ **Optimization Complete!**'),
                mo.md('<br>'),
                mo.md('---'),
                mo.md('<br>'),
            ]
        )

    _display
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## **3. Algorithm progress history**

    Assess convergence and features in the components. If needed, adjust the algorithm's parameters and rerun.
    """
    )
    return


@app.cell
def _(
    adv_iter,
    feature_map,
    get_algorithm_progress_plots,
    get_final_alphas_plot,
    history,
    mo,
    n_features,
):
    mo.stop(history is None, '')  # Show only after feature selector is run

    # Visualization of algorithm trajectories

    # Get progress plots
    progress_plots_dict = get_algorithm_progress_plots(
        history,
        elbo=True,
        mu=True,
        alpha=False,
        original_feature_names_mapping=feature_map,
        subsample_history_for_plotting=(
            True if ((adv_iter.value > 4000) or (n_features > 80)) else False
        ),
    )

    alphas_plots = get_final_alphas_plot(
        history,
        show_bar_plot=False,
        show_pie_chart=True,
    )
    alpha_piechart = alphas_plots[0]

    # Elbo convergence help text
    elbo_help = mo.accordion(
        {
            ' 📖 Guide': mo.md(
                r"""
                The [ELBO (Evidence Lower BOund)](https://en.wikipedia.org/wiki/Evidence_lower_bound) is the objective function that the algorithm maximizes, possibly combined with penalization: $ELBO - \lambda * penalty$

                The objective function's value should steadily increase and eventually plateau just below zero.
                Note that oscillations occur naturally due to inherent stochasticity.

                **What to look for:**
                - **Good convergence:** Steady upward trend that flattens into a plateau. The curve should stabilize before the end of iterations. The final 10-20% of iterations should show minimal change.
                - **Not converged:** Still increasing steeply at the end → increase number of iterations.
                - **Unstable:** Erratic oscillations are not dampened over time → decrease learning rate or adjust batch size.
                - **Too high absolute value:** Values in millions or more signify a major problem with the algorithm setup. Assess the features' trajectories below. (Values in low thousands, depending on lambda, are generally favorable.)
                """
            )
        }
    )

    # Feature convergence help text
    mu_help = mo.accordion(
        {
            ' 📖 Guide': mo.md(
                """
                **These plots are crucial for assessment of whether the feature selector produced sensible results.**

                Each plot shows how feature importance values (mu) evolve during optimization for one component.
                The algorithm assigns each feature to either the Spike (near zero) or Slab (non-zero) distribution.

                **What to look for:**
                - **Good separation:** Clear gap between features converging to ~0 and features with significant non-zero values.
                - **Optimization process:** 

                **Problem indicators:**
                - **All features → 0:** Over-regularization. Increase spike variance.
                - **Features' ordering does not change over time:** May indicate under-regularization.
                - **Too many non-zero features:** May indicate under-regularization. Carefully decrease spike variance and/or enforce stronger sparsity.
                - **Oscillating values:** Learning rate too high or batch size too small.
                - **Set of important features changes multiple times over time** (situation: while most features converge to 0, a few become nonzero. Then the nonzero features become 0 and others emerge. The set of important features significantly changes multiple times.): The algorithm cannot find any significant signal among noise. May indicate fundamental problem with the dataset.
                """
            )
        }
    )

    # Alpha help text
    alpha_help = mo.accordion(
        {
            ' 📖 Guide': mo.md(
                """
                The pie chart shows how the algorithm distributes probability mass across the components.
                (Alphas represent the mixing weights in the [Gaussian mixture model](https://en.wikipedia.org/wiki/Mixture_model) that approximate the [posterior distribution](https://en.wikipedia.org/wiki/Posterior_probability).)

                The alphas should correspond to the predictive potential of the components.

                **What to look for:**
                - **(Un)balanced distribution:** If you expect multiple solutions of comparable significance, alphas should be relatively balanced across components. If some components dominate, too many components may have been requested (i.e. the dataset supports fewer distinct solutions).
                """
            )
        }
    )

    mo.vstack(
        [
            mo.md('<br>'),
            mo.md('### 3.1 Objective function convergence'),
            elbo_help,
            progress_plots_dict['elbo'].update_layout(height=400, width=200 + adv_iter.value / 8),
            mo.md('<br>'),
            mo.md('### 3.2 Feature convergence in components'),
            mu_help,
            # Unpack all mu plots
            *[
                progress_plots_dict[_plot].update_layout(height=400, width=400 + adv_iter.value / 8)
                for _plot in progress_plots_dict.keys()
                if 'mu_' in _plot
            ],
            mo.md('<br>'),
            mo.md('### 3.3 Relative importances of components'),
            alpha_help,
            alpha_piechart.update_layout(height=450, width=450),
            mo.md('<br>'),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## **4. Recover solutions from components**

    Each component can be handled in multiple ways to yield feature sets = candidate solutions. Select your strategy.
    """
    )
    return


@app.cell
def _(df_processed, history, mo, sparsity_est):
    mo.stop(history is None, '')  # Show only after feature selector is run

    # checkboxes to pick solution types
    checkbox_out20_sol = mo.ui.checkbox(label='Outliers with STD > 2.0', value=True)
    checkbox_out25_sol = mo.ui.checkbox(label='Outliers with STD > 2.5', value=True)
    checkbox_out30_sol = mo.ui.checkbox(label='Outliers with STD > 3.0', value=True)
    checkbox_out35_sol = mo.ui.checkbox(label='Outliers with STD > 3.5', value=False)
    checkbox_top_sol = mo.ui.checkbox(label='Top few features', value=False)
    checkbox_full_sol = mo.ui.checkbox(label='All features with mu > threshold', value=False)

    # advanced settings
    top_n_features_selector = mo.ui.number(
        1,
        df_processed.shape[1],
        value=sparsity_est.value,
        label="Hard number of top features to be selected ('top' solution type).",
    )
    min_mu_selector = mo.ui.number(
        0,
        5,
        value=0.25,
        label="'Zero' threshold for mu values",
    )

    # display options
    checkbox_summary = mo.ui.checkbox(label='Summary', value=True, disabled=True)
    checkbox_matrix = mo.ui.checkbox(label='Feature distribution across components', value=True)
    checkbox_regression_l2 = mo.ui.checkbox(label='Regression with l2 regularization', value=True)
    checkbox_regression_l1 = mo.ui.checkbox(label='Regression with l1 regularization', value=False)

    # description of solution types
    solution_recovery_help = mo.accordion(
        {
            ' 📖 Guide': mo.md(
                """
                ### Solution recovery strategies

                Each component from the feature selector contains information about feature importance (mu values). These can be converted into feature sets (candidate solutions) using different strategies:

                **Outlier-based solutions** (recommended)

                Identifies features with unusually high |mu| values using [statistical outlier detection](https://en.wikipedia.org/wiki/Outlier). Features are selected if their |mu| exceeds:
                - **STD > 2.0**: Most inclusive (2 standard deviations from mean)
                - **STD > 2.5**: Moderately selective
                - **STD > 3.0**: More selective (common choice)
                - **STD > 3.5**: Most restrictive

                **Top-few solutions**

                Selects a fixed number of top features with highest |mu| values.
                The number is specified in the advanced settings below.

                **Thresholded solutions**

                Selects all features with |mu| values above an arbitrary threshold (defined in the advanced settings below).
                This approach is useful for further assessment of importance of the features selected in other strategies 
                because it can order all features in a component based on their |mu| values.
                For example, if a feature has a significant |mu| value in only one component with generously set threshold, it is likely noise.
                Important features usually have significant |mu| values in multiple components.
                """
            )
        }
    )

    # description of solution display options
    solution_display_help = mo.accordion(
        {
            ' 📖 Guide': mo.md(
                """
                For each solution type, you can display:

                **Summary** (always shown): Table listing all features in each component with their mu values, sorted by importance. Helps you understand which features the algorithm selected and their relative importance in the model.

                **Feature distribution across components**: Heatmap/matrix showing which features appear in which components. Useful for:
                - Identifying features that appear across multiple components (potentially robust predictors)
                - Checking solution diversity (do different components select different features?)
                - Understanding feature overlap patterns

                **Regression validation**: Quick, preliminary assessment of each solution's predictive potential using simple linear/logistic regression:
                - **L2 regularization** (Ridge): Handles correlated features well, generally more stable
                - **L1 regularization** (Lasso): Performs additional feature selection, may be more interpretable

                ⚠️ **Important:** These regression metrics use the *same data for training and testing* (no cross-validation), so they provide only a preliminary quality check. For rigorous evaluation, see the next step.

                The regression results help you quickly identify which solutions show promise before investing time in full model evaluation.
                """
            )
        }
    )

    mo.vstack(
        [
            mo.md('### 4.1 Pick solution types to be recovered from components:'),
            solution_recovery_help,
            mo.vstack(
                [
                    checkbox_out20_sol,
                    checkbox_out25_sol,
                    checkbox_out30_sol,
                    checkbox_out35_sol,
                    checkbox_top_sol,
                    checkbox_full_sol,
                ]
            ),
            mo.accordion(
                {'Advanced setting': mo.vstack([top_n_features_selector, min_mu_selector])}
            ),
            mo.md('<br>'),
            mo.md('### 4.2 Pick what is to be shown:'),
            solution_display_help,
            mo.vstack(
                [
                    checkbox_summary,
                    checkbox_matrix,
                    checkbox_regression_l2,
                    checkbox_regression_l1,
                ]
            ),
        ]
    )
    return (
        checkbox_full_sol,
        checkbox_matrix,
        checkbox_out20_sol,
        checkbox_out25_sol,
        checkbox_out30_sol,
        checkbox_out35_sol,
        checkbox_regression_l1,
        checkbox_regression_l2,
        checkbox_top_sol,
        min_mu_selector,
        top_n_features_selector,
    )


@app.cell
def _(history, mo):
    mo.stop(history is None, '')  # Show only after feature selector is run

    # The big RUN button
    recover_btn = mo.ui.run_button(label='Recover solutions', kind='success')

    mo.vstack(
        [
            mo.md('<br>'),
            recover_btn,
            mo.md('<br>'),
            mo.md('---'),
        ]
    )
    return (recover_btn,)


@app.cell
def _(
    checkbox_full_sol,
    checkbox_matrix,
    checkbox_out20_sol,
    checkbox_out25_sol,
    checkbox_out30_sol,
    checkbox_out35_sol,
    checkbox_regression_l1,
    checkbox_regression_l2,
    checkbox_top_sol,
    detect_task,
    df_processed,
    experiment_dir,
    feature_map,
    features_path_json,
    features_path_txt,
    get_features_from_solutions,
    get_features_in_components_plot,
    get_solution_summary_df,
    get_unique_features,
    history,
    min_mu_selector,
    mo,
    recover_btn,
    recover_solutions,
    save_feature_lists_json,
    save_feature_lists_txt,
    save_results,
    scaling_selector,
    solve_any_regression,
    top_n_features_selector,
    y,
):
    mo.stop(history is None, '')  # Show only after feature selector is run

    mo.stop(
        not recover_btn.value,
        mo.md('*Ready to recover solutions from components. Click button above.*'),
    )
    mo.stop(
        (top_n_features_selector.value is None) or (top_n_features_selector.value < 1),
        mo.md("Please set 'top few features' to value 1 or more."),
    )

    # Define which outliers are to be recovered
    outlier_deviation_thresholds = []
    if checkbox_out20_sol.value:
        outlier_deviation_thresholds.append(2.0)
    if checkbox_out25_sol.value:
        outlier_deviation_thresholds.append(2.5)
    if checkbox_out30_sol.value:
        outlier_deviation_thresholds.append(3.0)
    if checkbox_out35_sol.value:
        outlier_deviation_thresholds.append(3.5)

    # Recover solutions
    sol_full, sol_top, sol_outliers, _ = recover_solutions(
        search_history=history,
        desired_sparsity=top_n_features_selector.value,
        min_mu_threshold=min_mu_selector.value,
        original_feature_names_mapping=feature_map,
        use_median_for_outlier_detection=False,
        outlier_deviation_thresholds=outlier_deviation_thresholds,
    )

    # Put all the requested solution types into a single dictionary
    all_solutions = {}
    for _key, _outlier in sol_outliers.items():
        all_solutions[f'Outlier features ({" = ".join(_key.split(sep="_"))})'] = _outlier
    if checkbox_top_sol.value:
        all_solutions['Top features'] = sol_top
    if checkbox_full_sol.value:
        all_solutions['Thresholded features'] = sol_full

    mo.stop(
        all_solutions == {},
        mo.md('Cannot proceed. Please pick a solution type to recover.'),
    )

    # Extract which features are contained in which solution type
    # Get overviews and simple performance metrics
    solution_summary = {}  # one dateframe per solution type: feature names with mu values
    all_feature_sets = {}  # features per component, for each solution type
    unique_features_found = {}  # all unique features across all components of a solution type
    regression_metrics_l1 = {}
    regression_metrics_l2 = {}

    for _type, _solution in all_solutions.items():
        solution_summary[_type] = get_solution_summary_df(_solution)
        all_feature_sets[_type] = get_features_from_solutions(_solution)
        unique_features_found[_type] = get_unique_features(all_feature_sets[_type])

        # Quick validation with simple linear/logistic regression
        # l2-regularized
        if checkbox_regression_l2.value and (df_processed is not None):
            regression_metrics_l2[_type] = solve_any_regression(
                solutions=all_feature_sets[_type],
                df=df_processed,
                response=y,
                apply_scaling=scaling_selector.value,
                penalty='l2',
                verbose=False,
            )
        # l1-regularized
        if checkbox_regression_l1.value and (df_processed is not None):
            regression_metrics_l1[_type] = solve_any_regression(
                solutions=all_feature_sets[_type],
                df=df_processed,
                response=y,
                apply_scaling=scaling_selector.value,
                penalty='l1',
                verbose=False,
            )

    if save_results:
        # Save candidate solutions
        msg_features_json = save_feature_lists_json(all_feature_sets, features_path_json)
        msg_features_txt = save_feature_lists_txt(all_feature_sets, features_path_txt)

        # Stack all the outputs in the correct order
        _displays = [
            mo.md(f'📁 **All recovered solutions saved to:** `{experiment_dir}`'),
            mo.md(f'- {msg_features_txt.split("Candidate solutions saved to ")[1]}'),
            mo.md(f'- {msg_features_json.split("Candidate solutions saved to ")[1]}'),
            mo.md('---'),
            mo.md('<br><br>'),
        ]
    else:
        _displays = []

    for _type, _solution in all_solutions.items():
        # Get summary of a solution type
        _displays.append(mo.md(f'### Solution type: **{_type}**'))
        _displays.append(mo.ui.table(solution_summary[_type]))

        # Get a matrix of features vs. components
        if checkbox_matrix.value:
            _displays.append(
                get_features_in_components_plot(
                    solutions=all_feature_sets[_type],
                    features_to_show=unique_features_found[_type],
                ).update_layout(showlegend=False),
            )

        # Regression or classification?
        task_type = detect_task(y)

        # Get quick validation with a simple regression
        if checkbox_regression_l2.value or checkbox_regression_l1.value:
            regression_type = 'logistic' if task_type == 'classification' else 'linear'

        # l2-regularized
        if checkbox_regression_l2.value:
            _displays.append(
                mo.md(
                    f'#### **Quick l2-regularized {regression_type} regression validation** for {_type} (testing = training data):'
                )
            )
            _displays.append(mo.ui.table(regression_metrics_l2[_type]))

        # l1-regularized
        if checkbox_regression_l1.value:
            _displays.append(
                mo.md(
                    f'#### **Quick l1-regularized {regression_type} regression validation** for {_type} (testing = training data):'
                )
            )
            _displays.append(mo.ui.table(regression_metrics_l1[_type]))

        _displays.append(mo.md('<br><br>'))
        _displays.append(mo.md('---'))
        _displays.append(mo.md('<br><br>'))

    # Return all displays stacked vertically
    mo.vstack(_displays)
    return all_feature_sets, all_solutions, task_type, unique_features_found


@app.cell
def _(mo):
    mo.md(
        r"""
    ## **5. Modeling with candidate solutions**

    Evaluate discovered feature sets using nested cross-validation with scikit-learn models. This provides proper generalization performance estimates.
    """
    )
    return


@app.cell
def _(all_solutions, mo):
    # Stop if solutions not recovered
    mo.stop(
        all_solutions is None,
        mo.md('*Must recover solutions from components first. Click button above.*'),
    )

    # Select solution type for nested CV modeling
    radio_solutions_cv = mo.ui.radio(
        options=all_solutions.keys(),
        label='### 5.1 Choose one solution type to evaluate:',
    )

    modeling_help = mo.accordion(
        {
            '📖 About cross-validation': mo.md(
                """
                ### Why use cross-validation?

                Unlike the preliminary results above, this modeling provides cross-validation:
                - **Proper generalization estimates** through train/test splitting
                - **Unbiased performance metrics** that reflect real-world performance
                - **Multiple model options** beyond simple linear/logistic regression

                ### How it works

                - **Outer loop:** Splits data for performance evaluation
                - **Inner loop:** Fits models with hyperparameter tuning (implemented only for linear/logistic regressions, other models use default hyperparameters for speed)
                - **Result:** Metrics computed on held-out test data, aggregated over all outer folds
                
                This is the gold standard for evaluating predictive performance of discovered solutions, providing the most reliable assessment of their real-world utility.
                """
            )
        }
    )

    mo.vstack(
        [
            modeling_help,
            radio_solutions_cv,
        ]
    )
    return (radio_solutions_cv,)


@app.cell
def _(
    _get_available_models,
    all_solutions,
    mo,
    radio_solutions_cv,
    task_type,
):
    # Stop if solutions not recovered (task_type won't exist)
    mo.stop(
        all_solutions is None,
        output=mo.md('*Must recover solutions from components first.*<br><br>'),
    )

    mo.stop(
        radio_solutions_cv.value is None,
        output=mo.md('*Select a solution type above to continue.*<br><br>'),
    )

    # Get available models based on task type
    available_models = _get_available_models(task_type)

    # Create checkboxes for model selection with nice names
    nice_model_names = {
        'logistic_l2': 'Ridge regression',
        'logistic_l1': 'Lasso',
        'logistic_elasticnet': 'Elastic net',
        'linear_l2': 'Ridge regression',
        'linear_l1': 'Lasso',
        'linear_elasticnet': 'Elastic net',
        'decision_tree': 'Decision tree',
        'random_forest': 'Random forest',
        'xgboost': 'XGBoost',
        'svm': 'Support Vector Machine with RBF kernel',
        'knn': '3-Nearest Neighbors',
        'naive_bayes': 'Naive Bayes',
        'lda': 'Linear Discriminant Analysis',
        'qda': 'Quadratic Discriminant Analysis',
    }

    # Create reverse mapping for converting nice names back to technical names
    # Only include models available for this task type to avoid conflicts
    technical_model_names = {
        nice_model_names[model_name]: model_name for model_name in available_models
    }

    # Use mo.ui.dictionary to properly track checkbox state changes
    model_checkboxes = mo.ui.dictionary(
        {
            nice_model_names[model_name]: mo.ui.checkbox(
                value=True,
                label=nice_model_names[model_name],
            )
            for model_name in available_models
        }
    )

    # CV configuration
    cv_folds_selector = mo.ui.number(
        start=2,
        stop=20,
        value=5,
        step=1,
        label='Number of CV folds (outer loop)',
    )

    cv_loo_checkbox = mo.ui.checkbox(
        value=False,
        label='OR use Leave-One-Out CV instead (for small datasets)',
    )

    # Describe the modeling options with their default values
    help_models_regression = mo.accordion(
        {
            '📖 Guide': mo.md(
                """
                ### Regression Models

                **Ridge regression**
                - L2-regularized linear regression
                - Configuration: 5-fold cross-validation (RidgeCV)
                - Best for: simple baseline, robust to multicollinearity
                - Weakness: linear model

                **Lasso**
                - L1-regularized linear regression with
                - Configuration: 5-fold cross-validation for hyperparameter tuning
                - Best for: enforcing additional sparsity
                - Weakness: correlated features can cause instability, linear model

                **Elastic net**
                - Combined L1+L2 regularization with automatic hyperparameter tuning
                - Configuration: 5-fold cross-validation for hyperparameter tuning
                - Best for: balance between Ridge and Lasso
                - Weakness: may not perform well on highly correlated features

                **XGBoost**
                - Gradient boosting with regularization
                - Configuration: 100 estimators, learning_rate=0.1, max_depth=6
                - Best for: high performative complex modeling, handles missing data well
                - Weakness: can overfit, less interpretable

                **Support Vector Machine (SVM)**
                - SVM regression with RBF kernel
                - Configuration: C=1.0, gamma='scale' (SVR)
                - Best for: non-linear patterns
                - Weakness: sensitive to outliers, less interpretable

                **3-Nearest Neighbors (kNN)**
                - Instance-based learning using 3 nearest neighbors
                - Configuration: uniform weights
                - Best for: simple baseline, local patterns
                - Weakness: degrades in high dimensions, sensitive to irrelevant features

                **Decision tree**
                - Single decision tree with pruning
                - Configuration: max_depth=10, min_samples_split=2
                - Best for: interpretability, non-linear relationships
                - Weakness: can overfit, less stable
                
                **Random forest**
                - Ensemble of decision trees
                - Configuration: 100 trees, no max depth limit
                - Best for: non-linear relationships, feature interactions, robust to outliers
                - Weakness: can overfit on small datasets, less interpretable
                """
            )
        }
    )
    help_models_classification = mo.accordion(
        {
            '📖 Guide': mo.md(
                """
                ### Classification Models

                **Ridge regression**
                - L2-regularized linear regression
                - Configuration: 5-fold cross-validation (RidgeCV)
                - Best for: simple baseline, robust to multicollinearity
                - Weakness: linear model

                **Lasso**
                - L1-regularized linear regression with
                - Configuration: 5-fold cross-validation for hyperparameter tuning
                - Best for: enforcing additional sparsity
                - Weakness: correlated features can cause instability, linear model

                **Elastic net**
                - Combined L1+L2 regularization with automatic hyperparameter tuning
                - Configuration: 5-fold cross-validation for hyperparameter tuning
                - Best for: balance between Ridge and Lasso
                - Weakness: may not perform well on highly correlated features

                **XGBoost**
                - Gradient boosting with regularization
                - Configuration: 100 estimators, learning_rate=0.1, max_depth=6
                - Best for: high performative complex modeling, handles missing data well
                - Weakness: can overfit, less interpretable

                **Support Vector Machine (SVM)**
                - SVM regression with RBF kernel
                - Configuration: C=1.0, gamma='scale' (SVR)
                - Best for: non-linear patterns
                - Weakness: sensitive to outliers, less interpretable

                **3-Nearest Neighbors (kNN)**
                - Instance-based learning using 3 nearest neighbors
                - Configuration: uniform weights
                - Best for: simple baseline, local patterns
                - Weakness: degrades in high dimensions, sensitive to irrelevant features

                **Decision tree**
                - Single decision tree with pruning
                - Configuration: max_depth=10, min_samples_split=2
                - Best for: interpretability, non-linear relationships
                - Weakness: can overfit, less stable
                
                **Random forest**
                - Ensemble of decision trees
                - Configuration: 100 trees, no max depth limit
                - Best for: non-linear relationships, feature interactions, robust to outliers
                - Weakness: can overfit on small datasets, less interpretable

                **Naive Bayes**
                - Gaussian Naive Bayes classifier
                - Best for: simple baseline, small sample size
                - Weakness: strong independence assumption, may underperform on complex datasets

                **Linear Discriminant Analysis (LDA)**
                - Linear classifier assuming Gaussian distributions
                - Best for: multi-class problems, dimensionality reduction
                - Weakness: assumes equal covariance matrices, may underperform on non-linear boundaries

                **Quadratic Discriminant Analysis (QDA)**
                - Quadratic classifier assuming Gaussian distributions with different covariances
                - Best for: non-linear decision boundaries, different class covariances
                - Weakness: sensitive to outliers, may overfit on small datasets
                """
            )
        }
    )

    if task_type == 'regression':
        help_models = help_models_regression
    elif task_type == 'classification':
        help_models = help_models_classification

    # Model selection UI
    model_selection_ui = mo.vstack(
        [
            mo.md('### 5.2 Select models to evaluate:'),
            mo.md(f'*{task_type.capitalize()}: {len(available_models)} available models*'),
            help_models,
        ]
        + [model_checkboxes[nice_model_names[model_name]] for model_name in available_models]
    )

    cv_config_ui = mo.accordion(
        {
            'Cross-validation settings': mo.vstack(
                [
                    cv_folds_selector,
                    cv_loo_checkbox,
                ]
            )
        }
    )

    mo.vstack(
        [
            mo.md('<br>'),
            model_selection_ui,
            cv_config_ui,
        ]
    )
    return (
        available_models,
        cv_folds_selector,
        cv_loo_checkbox,
        model_checkboxes,
        technical_model_names,
    )


@app.cell
def _(mo, model_checkboxes, radio_solutions_cv, technical_model_names):
    mo.stop(
        radio_solutions_cv.value is None,
        output=mo.md('*Select solution type first.*'),
    )

    # Check if at least one model is selected
    # model_checkboxes.value returns a dict of {nice_name: True/False}
    # Convert nice names back to technical names for evaluation
    selected_models = [
        technical_model_names[nice_name]
        for nice_name, is_checked in model_checkboxes.value.items()
        if is_checked
    ]

    mo.stop(
        len(selected_models) == 0,
        output=mo.md('*Please select at least one model to evaluate.*'),
    )

    # Run button for nested CV modeling
    run_cv_btn = mo.ui.run_button(
        label=f'Run modeling ({len(selected_models)} model{"s" if len(selected_models) > 1 else ""})',
        kind='success',
    )

    mo.vstack(
        [
            mo.md('<br>'),
            run_cv_btn,
            mo.md('---'),
        ]
    )
    return run_cv_btn, selected_models


@app.cell
def _(
    all_feature_sets,
    all_solutions,
    cv_folds_selector,
    cv_loo_checkbox,
    df_processed,
    evaluate_all_solutions,
    experiment_dir,
    mo,
    nice_model_names,
    pd,
    radio_solutions_cv,
    run_cv_btn,
    save_results,
    selected_models,
    scaling_selector,
    stratification_col_selector,
    stratify_col,
    task_type,
    y,
):
    mo.stop(
        not run_cv_btn.value,
        output=mo.md('*Ready to run modeling. Press button above.*'),
    )

    # Get the selected solution type
    selected_solution_type_cv = radio_solutions_cv.value
    selected_solution_cv = all_solutions[selected_solution_type_cv]
    _solution_name = selected_solution_type_cv.replace(' ', '_').lower()

    # Determine CV folds
    cv_folds = 'loo' if cv_loo_checkbox.value else cv_folds_selector.value

    # Determine stratification description for CV display
    if stratify_col is not None:
        _strat_info = f'Custom (column: {stratification_col_selector.value})'
    else:
        _strat_info = 'Default: by target' if task_type == 'classification' else 'None'

    _cv_displays = []
    _cv_displays.append(
        mo.md(
            f"""
            ### 5.3 Evaluating solution: **{selected_solution_type_cv}**
            - Number of feature sets: **{len(selected_solution_cv)}**
            - Models: **{', '.join([nice_model_names.get(m, m) for m in selected_models])}**
            - Scaling: **{scaling_selector.value or 'None'}**
            - Outer CV type: **{'Leave-One-Out' if cv_folds == 'loo' else f'{cv_folds}-fold'}**
            - Stratification: **{_strat_info}**
            
            """
        )
    )
    _cv_displays.append(mo.md('<br>'))

    # Get feature sets for this solution type
    component_features_cv = all_feature_sets[selected_solution_type_cv]

    # Evaluate with each selected model
    all_cv_results = {}
    for model_name in selected_models:
        _cv_displays.append(mo.md(f'### Model: **{nice_model_names.get(model_name, model_name)}**'))

        # Run nested CV evaluation
        cv_results = evaluate_all_solutions(
            solutions=component_features_cv,
            df=df_processed,
            response=y,
            model_name=model_name,
            apply_scaling=scaling_selector.value,
            outer_cv_folds=cv_folds,
            random_state=42,
            verbose=False,
            use_markdown=False,
            stratify=stratify_col,
        )

        all_cv_results[model_name] = cv_results

        # Display results for this model
        if not cv_results.empty:
            _cv_displays.append(mo.ui.table(cv_results))

            # Save results if enabled
            if save_results:
                _results_path = f'{experiment_dir}/modeling_{model_name}_{_solution_name}.csv'
                cv_results.to_csv(_results_path)
                _cv_displays.append(mo.md(f'📁 Saved to: `{_results_path.split("/")[-1]}`'))
        else:
            _cv_displays.append(mo.md('⚠️ No solutions could be evaluated (insufficient samples).'))

        _cv_displays.append(mo.md('<br>'))

    # Model comparison if multiple models
    if len(selected_models) > 1 and all(not df.empty for df in all_cv_results.values()):
        # Create comparison table
        comparison_data = []
        for solution_name in all_cv_results[selected_models[0]].index:
            row = {'Solution': solution_name}
            for model_name in selected_models:
                if solution_name in all_cv_results[model_name].index:
                    # Get primary metric (store as float for styling)
                    if 'f1_score' in all_cv_results[model_name].columns:
                        metric_val = all_cv_results[model_name].loc[solution_name, 'f1_score']
                        row[model_name] = metric_val
                        primary_metric = 'f1_score'
                    elif 'r2_score' in all_cv_results[model_name].columns:
                        metric_val = all_cv_results[model_name].loc[solution_name, 'r2_score']
                        row[model_name] = metric_val
                        primary_metric = 'r2_score'
            comparison_data.append(row)

        comparison_df = pd.DataFrame(comparison_data)

        # Apply yellow gradient highlighting with custom function
        def highlight_values(val, vmin, vmax):
            """Apply white-to-yellow background based on value"""
            if pd.isna(val) or not isinstance(val, (int, float)):
                return ''
            # Normalize value to 0-1 range
            norm_val = (val - vmin) / (vmax - vmin) if vmax > vmin else 0.5
            # Create white to light yellow gradient: higher values = lighter yellow
            # RGB: white (255, 255, 255) to light yellow (255, 255, 150)
            r = 255  # stays at 255
            g = 255  # stays at 255
            b = int(255 - 105 * norm_val)  # 255 to 150 (lighter yellow)
            return f'background-color: rgb({r}, {g}, {b})'

        # Get min/max for normalization
        numeric_cols = [col for col in comparison_df.columns if col != 'Solution']
        vmin = comparison_df[numeric_cols].min().min()
        vmax = comparison_df[numeric_cols].max().max()

        # Apply styling
        styled_comparison = comparison_df.style.map(
            lambda val: highlight_values(val, vmin, vmax), subset=numeric_cols
        ).format({col: '{:.3f}' for col in numeric_cols})

        _cv_displays.append(mo.md('### 📊 5.4 Model comparison'))
        _cv_displays.append(
            mo.md(f'Best performance for each solution across all models (by {primary_metric}):')
        )
        # Render styled dataframe as HTML
        _cv_displays.append(mo.Html(styled_comparison.to_html()))

        # if comparison_df is not empty, save it to a file
        if not comparison_df.empty and save_results:
            _comparison_path = f'{experiment_dir}/model_comparison_{_solution_name}.csv'
            comparison_df.to_csv(_comparison_path, index=False)
            _cv_displays.append(
                mo.md(f'📁 Model comparison saved to: `{_comparison_path.split("/")[-1]}`')
            )

    _cv_displays.append(mo.md('<br>'))
    _cv_displays.append(mo.md('---'))
    _cv_displays.append(mo.md('<br>'))

    mo.vstack(_cv_displays)
    return


@app.cell
def _():
    return


if __name__ == '__main__':
    app.run()
