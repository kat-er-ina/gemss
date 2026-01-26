"""
Benchmark script for BayesianFeatureSelector.optimize.

Runs a configurable optimization loop on synthetic data and reports timing and
basic ELBO statistics. Intended for iterative performance tuning.
"""

from __future__ import annotations

import argparse
import cProfile
import json
import pstats
import time
from typing import Any, Dict, List, cast

import numpy as np
import torch

import gemss.config as C
from gemss.data_handling.generate_artificial_dataset import generate_artificial_dataset
from gemss.feature_selection.inference import BayesianFeatureSelector


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _filter_cprofile_stats(stats: pstats.Stats, min_calls: int) -> None:
    if min_calls <= 1:
        return

    stats_any = cast(Any, stats)
    stats_any.stats = {
        func: stat for func, stat in stats_any.stats.items() if stat[1] >= min_calls
    }
    stats_any.total_calls = 0
    stats_any.prim_calls = 0
    stats_any.total_tt = 0
    stats_any.max_name_len = 0
    stats_any.top_level = set()
    stats_any.all_callees = None
    stats_any.fcn_list = None
    stats_any.get_top_level_stats()


def _make_dataset(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    # Seed numpy so nan masking in generate_artificial_dataset is reproducible.
    np.random.seed(args.dataset_seed)
    df, y, _, _ = generate_artificial_dataset(
        n_samples=args.n_samples,
        n_features=args.n_features,
        n_solutions=args.n_solutions,
        sparsity=args.sparsity,
        noise_data_std=args.noise_std,
        nan_ratio=args.nan_ratio,
        binarize=args.binarize,
        binary_response_ratio=args.binary_response_ratio,
        random_seed=args.dataset_seed,
        save_to_csv=args.save_dataset,
        print_data_overview=args.print_data_overview,
        show_feature_correlations=args.show_feature_correlations,
    )

    X = df.to_numpy(dtype=np.float32, copy=False)
    y_values = y.to_numpy(dtype=np.float32, copy=False)
    return X, y_values


def _run_once(
    args: argparse.Namespace,
    seed: int,
    enable_torch_profile: bool,
    X: np.ndarray,
    y: np.ndarray,
) -> Dict[str, Any]:
    np.random.seed(seed)
    torch.manual_seed(seed)
    n_features = X.shape[1]

    selector = BayesianFeatureSelector(
        n_features=n_features,
        n_components=args.n_components,
        X=X,
        y=y,
        prior=args.prior,
        sss_sparsity=args.sss_sparsity,
        sample_more_priors_coeff=args.sample_more_priors_coeff,
        var_slab=args.var_slab,
        var_spike=args.var_spike,
        weight_slab=args.weight_slab,
        weight_spike=args.weight_spike,
        student_df=args.student_df,
        student_scale=args.student_scale,
        lr=args.lr,
        batch_size=args.batch_size,
        n_iter=args.n_iter,
        device=args.device,
    )

    start = time.perf_counter()
    if enable_torch_profile:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if args.device == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
        ) as prof:
            history = selector.optimize(
                regularize=args.regularize,
                lambda_jaccard=args.lambda_jaccard,
                verbose=args.verbose,
            )
        print(
            prof.key_averages().table(
                sort_by="self_cpu_time_total",
                row_limit=args.torch_profile_row_limit,
            )
        )
        if args.torch_profile_output:
            prof.export_chrome_trace(args.torch_profile_output)
    else:
        history = selector.optimize(
            regularize=args.regularize,
            lambda_jaccard=args.lambda_jaccard,
            verbose=args.verbose,
        )
    elapsed = time.perf_counter() - start

    elbo = np.array(history["elbo"], dtype=float)
    tail_k = min(args.tail_k, len(elbo)) if len(elbo) else 0
    mean_tail = float(elbo[-tail_k:].mean()) if tail_k else None

    return {
        "elapsed_s": elapsed,
        "time_per_iter_s": elapsed / max(args.n_iter, 1),
        "final_elbo": float(elbo[-1]) if len(elbo) else None,
        "mean_elbo_tail": mean_tail,
    }


def _summarize_runs(run_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    times = np.array([run["elapsed_s"] for run in run_results], dtype=float)
    per_iter = np.array([run["time_per_iter_s"] for run in run_results], dtype=float)

    return {
        "runs": len(run_results),
        "elapsed_s_mean": float(times.mean()),
        "elapsed_s_std": float(times.std(ddof=1)) if len(times) > 1 else 0.0,
        "time_per_iter_s_mean": float(per_iter.mean()),
        "time_per_iter_s_std": float(per_iter.std(ddof=1))
        if len(per_iter) > 1
        else 0.0,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark BayesianFeatureSelector.optimize on synthetic data."
    )

    parser.add_argument("--n-samples", type=_positive_int, default=C.N_SAMPLES)
    parser.add_argument("--n-features", type=_positive_int, default=C.N_FEATURES)
    parser.add_argument(
        "--n-solutions",
        "--n-generating-solutions",
        dest="n_solutions",
        type=_positive_int,
        default=C.N_GENERATING_SOLUTIONS,
    )
    parser.add_argument("--sparsity", type=_positive_int, default=C.SPARSITY)
    parser.add_argument("--n-components", type=_positive_int, default=C.N_CANDIDATE_SOLUTIONS)
    parser.add_argument("--n-iter", type=_positive_int, default=C.N_ITER)
    parser.add_argument("--batch-size", type=_positive_int, default=C.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=C.LEARNING_RATE)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")

    parser.add_argument(
        "--prior", choices=["ss", "sss", "student"], default=C.PRIOR_TYPE
    )
    default_sss_sparsity = C.PRIOR_SPARSITY if C.PRIOR_SPARSITY is not None else 3
    parser.add_argument("--sss-sparsity", type=_positive_int, default=default_sss_sparsity)
    parser.add_argument(
        "--sample-more-priors-coeff", type=float, default=C.SAMPLE_MORE_PRIORS_COEFF
    )
    parser.add_argument("--var-slab", type=float, default=C.VAR_SLAB)
    parser.add_argument("--var-spike", type=float, default=C.VAR_SPIKE)
    parser.add_argument("--weight-slab", type=float, default=C.WEIGHT_SLAB)
    parser.add_argument("--weight-spike", type=float, default=C.WEIGHT_SPIKE)
    parser.add_argument("--student-df", type=float, default=C.STUDENT_DF)
    parser.add_argument("--student-scale", type=float, default=C.STUDENT_SCALE)

    parser.add_argument(
        "--regularize",
        action=argparse.BooleanOptionalAction,
        default=C.IS_REGULARIZED,
    )
    parser.add_argument("--lambda-jaccard", type=float, default=C.LAMBDA_JACCARD)

    parser.add_argument("--noise-std", type=_non_negative_float, default=C.NOISE_STD)
    parser.add_argument(
        "--nan-ratio",
        "--missing-rate",
        dest="nan_ratio",
        type=_non_negative_float,
        default=C.NAN_RATIO,
    )
    parser.add_argument(
        "--binarize",
        action=argparse.BooleanOptionalAction,
        default=C.BINARIZE,
    )
    parser.add_argument(
        "--binary-response-ratio",
        type=_non_negative_float,
        default=C.BINARY_RESPONSE_RATIO,
    )
    parser.add_argument("--dataset-seed", type=int, default=C.DATASET_SEED)
    parser.add_argument("--save-dataset", action="store_true")
    parser.add_argument("--print-data-overview", action="store_true")
    parser.add_argument("--show-feature-correlations", action="store_true")

    parser.add_argument("--runs", type=_positive_int, default=1)
    parser.add_argument("--seed", type=int, default=C.DATASET_SEED)
    parser.add_argument("--tail-k", type=_positive_int, default=10)

    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--cprofile-output", type=str, default=None)
    parser.add_argument("--cprofile-sort", type=str, default="cumtime")
    parser.add_argument("--cprofile-lines", type=_positive_int, default=30)
    parser.add_argument("--cprofile-min-calls", type=_positive_int, default=10)
    parser.add_argument("--torch-profile", action="store_true")
    parser.add_argument("--torch-profile-output", type=str, default=None)
    parser.add_argument("--torch-profile-row-limit", type=_positive_int, default=25)

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but not available on this machine.")

    if args.nan_ratio > 1.0:
        raise SystemExit("nan-ratio must be between 0 and 1.")
    if args.binary_response_ratio > 1.0:
        raise SystemExit("binary-response-ratio must be between 0 and 1.")

    profiler = None
    if args.cprofile_output:
        profiler = cProfile.Profile()
        profiler.enable()

    X, y = _make_dataset(args)
    run_results = []
    for idx in range(args.runs):
        run_seed = args.seed + idx
        enable_torch_profile = args.torch_profile and idx == 0
        run_results.append(_run_once(args, run_seed, enable_torch_profile, X, y))

    if profiler is not None:
        profiler.disable()
        profiler.dump_stats(args.cprofile_output)
        stats = pstats.Stats(profiler)
        _filter_cprofile_stats(stats, args.cprofile_min_calls)
        stats.sort_stats(args.cprofile_sort)
        stats.print_stats(args.cprofile_lines)

    summary = _summarize_runs(run_results)

    payload = {
        "config": vars(args),
        "summary": summary,
        "runs": run_results,
    }

    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=True)

    print(
        "Benchmark results",
        f"- runs: {summary['runs']}",
        f"- elapsed mean: {summary['elapsed_s_mean']:.4f}s "
        f"(std {summary['elapsed_s_std']:.4f}s)",
        f"- time/iter mean: {summary['time_per_iter_s_mean']:.6f}s "
        f"(std {summary['time_per_iter_s_std']:.6f}s)",
        sep="\n",
    )


if __name__ == "__main__":
    main()
