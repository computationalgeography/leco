# pyright: reportAttributeAccessIssue=false, reportArgumentType=false

import argparse
import ast
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from scipy import stats

# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------


def save_fig(fig: Figure, name: str, output_dir: Path) -> None:
    """Save figure and close it."""
    fig.savefig(output_dir / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {name}")


def base_plot(
    ax: Axes,
    agg: pd.DataFrame,
    x_col: str,
    baseline_x: float,
    ylabel: str,
    variable: str,
    parameter_labels: dict,
) -> None:
    """Shared scatter plot logic: median (circle) + min/max (triangles) + baseline vline."""
    ax.axvline(baseline_x, color="darkgrey", linestyle="--", linewidth=1.3)
    ax.scatter(agg[x_col], agg["median_val"], s=80, color="black", zorder=3, label="median")
    ax.scatter(agg[x_col], agg["min_val"], s=80, color="#01204E", zorder=3, marker="^", label="Min")
    ax.scatter(agg[x_col], agg["max_val"], s=80, color="#98C1D9", zorder=3, marker="^", label="Max")
    ax.set_xlabel(parameter_labels.get(variable, variable), fontsize=24)
    ax.set_ylabel(ylabel, fontsize=24)
    ax.tick_params(axis="both", labelsize=20)
    ax.legend(fontsize=16)
    ax.grid(True, alpha=0.3)


# ---------------------------------------------------------------------------
# Shared plots (used for both baseline and sensitivity)
# ---------------------------------------------------------------------------


def filter_baseline(df: pd.DataFrame, baseline_values: dict | None) -> pd.DataFrame:
    """Regulate different input.

    If baseline_values is provided, filters to that parameter combination first
    (sensitivity mode). Otherwise uses all rows (baseline mode).
    """
    if baseline_values is not None:
        mask = pd.Series([True] * len(df), index=df.index)
        for variable, val in baseline_values.items():
            mask &= df[variable] == val
        df = cast("pd.DataFrame", df[mask])

    return df


def plot_seed_variance_base(
    df: pd.DataFrame,
    output_dir: Path,
    baseline_values: dict | None = None,
) -> None:
    """Plot language richness per seed over time.

    If baseline_values is provided, filters to that parameter combination first
    (sensitivity mode). Otherwise uses all rows (baseline mode).
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    df_plot = df.copy()

    df_plot = filter_baseline(df_plot, baseline_values)

    seeds = sorted(df_plot["seed"].unique())
    colors = plt.get_cmap("viridis")(np.linspace(0, 1, len(seeds)))

    for i, (seed, color) in enumerate(zip(seeds, colors, strict=True)):
        seed_df = df_plot[df_plot["seed"] == seed]
        grouped = seed_df.groupby("step")["language_count"].median()
        ax.plot(grouped.index, grouped.values, color=color, alpha=0.7, linewidth=1, label=f"Run {i + 1}")

    ax.set_xlabel("Time steps", size=22)
    ax.set_ylabel("Number of Languages", size=22)
    ax.tick_params(axis="both", labelsize=18)
    leg = ax.legend(loc="upper right", fontsize=16)

    for legobj in leg.legend_handles:
        legobj.set_linewidth(3.0)  # type: ignore[arg-type]
    fig.tight_layout()

    fig.savefig(output_dir / "seed_variance_base.pdf", dpi=500, bbox_inches="tight")
    plt.close()
    print("Saved seed_variance_base.pdf")


def plot_speaker_dist_base(
    df: pd.DataFrame,
    output_dir: Path,
    baseline_values: dict | None = None,
    bin_number: int = 10,
) -> None:
    """Plot median speaker distribution for the last time step, averaged across seeds.

    If baseline_values is provided, filters to that parameter combination first
    (sensitivity mode). Otherwise uses all rows (baseline mode).
    """
    df = filter_baseline(df, baseline_values)

    # Select final time step
    last_step = df["year"].max()
    df_last = df[(df["year"] == last_step) & (df["speakers"].notna())]
    seeds = sorted(df_last["seed"].unique())

    # Collect all speakers per seed
    speakers_by_seed = {}
    for seed in seeds:
        df_seed = df_last[df_last["seed"] == seed]
        speakers_by_seed[seed] = np.concatenate(
            df_seed["speakers"].apply(ast.literal_eval).values,
        )

    # Find the maximum number of speakers
    max_val = max(arr.max() for arr in speakers_by_seed.values())

    # Divide the log-transformed speakers in even bins
    bins = np.logspace(np.log10(1), np.log10(max_val + 1), num=bin_number)

    # Compute histograms for each seed
    histograms = []
    for seed in seeds:
        hist, _ = np.histogram(speakers_by_seed[seed], bins=bins)
        histograms.append(hist)
    histograms = np.array(histograms)

    # Get the median, minimum and maximum
    median_hist = np.median(histograms, axis=0)
    minimum = np.min(histograms, axis=0)
    maximum = np.max(histograms, axis=0)

    ## Normal reference curve based on median and standard deviation of the log-transformed data
    # Collect the median and standard deviation per seed
    medians, sigmas = [], []
    for seed in seeds:
        log_speakers_seed = np.log10(speakers_by_seed[seed])
        medians.append(np.median(log_speakers_seed))
        sigmas.append(log_speakers_seed.std())

    # Get the general median and standard deviation
    median = np.median(medians)
    sigma = np.median(sigmas)

    # Median number of languages per seed, used to scale curve height to match histogram counts
    n_median = np.median(
        [len(v) for v in speakers_by_seed.values()],
    )  # median languages per seed per speaker number
    bin_width_log = np.diff(np.log10(bins)).mean()

    # Smooth curve spanning the same range as the histogram bins
    x_smooth_log = np.linspace(np.log10(bins[0]), np.log10(bins[-1]), 300)
    pdf_vals = stats.norm.pdf(x_smooth_log, median, sigma)
    expected_counts_smooth = pdf_vals * bin_width_log * n_median
    x_smooth = 10**x_smooth_log

    fig, ax = plt.subplots(figsize=(10, 6))

    # Extend arrays to bin edges so the step line reaches the last bin's right edge
    median_for_step = np.append(median_hist, median_hist[-1])
    min_for_step = np.append(minimum, minimum[-1])
    max_for_step = np.append(maximum, maximum[-1])

    # median line
    ax.step(
        bins,
        median_for_step,
        where="post",
        color="darkorange",
        linewidth=2,
        label="Median",
    )

    # IQR/min-max band
    ax.fill_between(
        bins,
        min_for_step,
        max_for_step,
        step="post",
        color="darkorange",
        alpha=0.15,
        label="Min/Max range",
    )

    # log-normal curve
    ax.plot(
        x_smooth,
        expected_counts_smooth,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label="Normal distribution reference",
    )

    ax.set_xscale("log")
    ax.set_xlabel("Number of Agents Speaking a Language (log-transformed)", size=22)
    ax.set_ylabel("Number of Languages", size=22)
    ax.tick_params(axis="both", labelsize=18)
    ax.legend(fontsize=16, loc="upper left")
    fig.tight_layout()
    save_fig(fig, "speaker_dist_baseline_median.pdf", output_dir)


# ---------------------------------------------------------------------------
# Baseline-only: summary statistics CSV + plots
# ---------------------------------------------------------------------------

BASELINE_METRICS_TEMPORAL = [
    "language_count",
    "divergences",
    "convergences",
    "speaker_median",
    "speaker_min",
    "speaker_max",
    "nncor_unnormalized",
    "nncor_normalized",
    "simpson_baseline",
]

BASELINE_METRICS_FINAL = [
    "family_count",
    "languages_per_family_median",
    "languages_per_family_min",
    "languages_per_family_max",
    "speakers_per_family_median",
    "speakers_per_family_min",
    "speakers_per_family_max",
]

BASELINE_METRICS = BASELINE_METRICS_TEMPORAL + BASELINE_METRICS_FINAL

METRIC_LABELS = {
    "language_count": "Number of Languages",
    "divergences": "Divergences",
    "convergences": "Convergences",
    "speaker_median": "Speakers per language (median)",
    "speaker_min": "Speakers per language (min)",
    "speaker_max": "Speakers per language (max)",
    "nncor_unnormalized": "Spatial clustering (unnormalized)",
    "nncor_normalized": "Spatial clustering (normalized)",
    "simpson_baseline": "Simpson baseline",
    "family_count": "Number of language families",
    "languages_per_family_median": "Languages per family (median)",
    "languages_per_family_min": "Languages per family (min)",
    "languages_per_family_max": "Languages per family (max)",
    "speakers_per_family_median": "Speakers per family (median)",
    "speakers_per_family_min": "Speakers per family (min)",
    "speakers_per_family_max": "Speakers per family (max)",
}


def write_baseline_summary(df: pd.DataFrame, output_dir: Path) -> None:
    """Write two CSVs.

    1. Per-seed, per-step values for the last N time steps.
    2. Aggregated median/min/max across seeds and those last N steps.
    """
    # Only select the final time step
    final_year = df["year"].max()
    df_last = df[df["year"].isin([final_year])].copy()

    # --- Per-seed, per-step CSV ---
    per_seed_cols = ["seed"] + [m for m in BASELINE_METRICS if m in df_last.columns]

    df_per_seed = cast("pd.DataFrame", df_last[per_seed_cols]).sort_values(by=["seed"]).reset_index(drop=True)

    seeds = sorted(df_per_seed["seed"].unique())
    wide_rows = []
    for metric in BASELINE_METRICS:
        if metric not in df_per_seed.columns:
            continue
        row = {"metric": metric, "label": METRIC_LABELS.get(metric, metric)}
        for seed in seeds:
            val = df_per_seed[df_per_seed["seed"] == seed][metric].iloc[0]
            row[f"seed_{seed}"] = val
        wide_rows.append(row)

    wide_df = pd.DataFrame(wide_rows)
    wide_path = output_dir / "baseline_summary_per_seed.csv"
    wide_df.to_csv(wide_path, index=False)
    print(f"Saved {wide_path}")

    # --- Aggregated CSV: median/min/max across seeds and last N steps ---
    agg_rows = []
    for metric in BASELINE_METRICS:
        if metric not in df_last.columns:
            continue
        values = df_last[metric].dropna()
        agg_rows.append(
            {
                "metric": metric,
                "label": METRIC_LABELS.get(metric, metric),
                "median": values.median(),
                "min": values.min(),
                "max": values.max(),
                "n_observations": len(values),
            },
        )
    agg_df = pd.DataFrame(agg_rows)
    agg_path = output_dir / "baseline_summary_aggregated.csv"
    agg_df.to_csv(agg_path, index=False)
    print(f"Saved {agg_path}")


def plot_baseline_metrics_over_time(df: pd.DataFrame, output_dir: Path) -> None:
    """Plot each baseline metric over time: median across seeds with minimum/maximum band."""
    metrics = [m for m in BASELINE_METRICS if m in df.columns]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 4 * len(metrics)), sharex=True)

    if len(metrics) == 1:
        axes = [axes]

    seeds = sorted(df["seed"].unique())
    colors = plt.get_cmap("viridis")(np.linspace(0, 1, len(seeds)))

    for ax, metric in zip(axes, metrics, strict=True):
        # Per-seed lines
        for seed, color in zip(seeds, colors, strict=True):
            seed_df = df[df["seed"] == seed].groupby("step")[metric].median()
            ax.plot(seed_df.index, seed_df.values, color=color, alpha=0.5, linewidth=1)

        # median and min/max across seeds
        grouped = df.groupby("step")[metric].agg(["median", "min", "max"]).reset_index()
        ax.plot(grouped["step"], grouped["median"], color="black", linewidth=2, label="median")
        ax.fill_between(
            grouped["step"],
            grouped["min"],
            grouped["max"],
            alpha=0.2,
            color="black",
            label="Min/Max",
        )

        ax.set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=13)
        ax.tick_params(axis="both", labelsize=11)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)

    def format_kyears(x: float, _: int | None) -> str:
        return f"{int(x / 50)}"

    axes[-1].xaxis.set_major_formatter(ticker.FuncFormatter(format_kyears))
    axes[-1].set_xlabel("Time steps", fontsize=20)
    axes[-1].xaxis.set_major_locator(ticker.MultipleLocator(base=20000))

    fig.suptitle("Baseline run metrics over time", fontsize=16, y=1.01)
    fig.tight_layout()
    save_fig(fig, "baseline_metrics_over_time.pdf", output_dir)


def plot_baseline_boxplots(df: pd.DataFrame, output_dir: Path) -> None:
    """Boxplot per metric across seeds for the last N time steps."""
    # Only select the final time step
    final_year = df["year"].max()
    df_last = df[df["year"].isin([final_year])].copy()

    metrics = [m for m in BASELINE_METRICS if m in df_last.columns]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 5))

    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics, strict=True):
        seeds = sorted(df_last["seed"].unique())
        data_per_seed = [df_last[df_last["seed"] == s][metric].dropna().to_numpy() for s in seeds]
        ax.boxplot(data_per_seed, tick_labels=[f"S{s}" for s in seeds], patch_artist=True)
        ax.set_title(METRIC_LABELS.get(metric, metric), fontsize=11)
        ax.tick_params(axis="x", labelsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Metric distributions per seed", fontsize=13)
    fig.tight_layout()
    save_fig(fig, "baseline_boxplots.pdf", output_dir)


def create_baseline_plots(input_path: Path) -> None:
    """Entry point for baseline mode."""
    df = pd.read_csv(input_path)
    output_dir = input_path.parent / "baseline_plots"
    output_dir.mkdir(exist_ok=True)

    write_baseline_summary(df, output_dir)
    plot_seed_variance_base(df, output_dir, baseline_values=None)
    plot_speaker_dist_base(df, output_dir, baseline_values=None)
    plot_baseline_metrics_over_time(df, output_dir)
    plot_baseline_boxplots(df, output_dir)


# ---------------------------------------------------------------------------
# Classification comparison plots
# ---------------------------------------------------------------------------


def plot_classification_comparison(
    baseline_path: Path,
    classification_files: list[tuple[str, Path]],
    output_dir: Path,
) -> None:
    """Plot language richness over time for different classification methods.

    Shows median with min/max shaded area per method, plus the baseline.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ["tab:brown", "tab:pink", "tab:olive", "tab:cyan"]

    def plot_method(df: pd.DataFrame, label: str, color: tuple, style: str = "-") -> None:
        grouped = df.groupby("step")["language_count"].agg(["median", "min", "max"]).reset_index()
        ax.plot(grouped["step"], grouped["median"], color=color, linestyle=style, linewidth=2, label=label)
        ax.fill_between(grouped["step"], grouped["min"], grouped["max"], alpha=0.15, color=color)

    # Plot baseline first
    df_baseline = pd.read_csv(baseline_path)
    plot_method(df_baseline, "Baseline", "black")

    # Plot each classification method
    for (name, path), color in zip(classification_files, colors[0:], strict=False):
        df = pd.read_csv(path)
        plot_method(df, name, color)

    handles, labels = ax.get_legend_handles_labels()
    ci_patch = Patch(facecolor="gray", alpha=0.4, label="Min/Max range")
    handles.append(ci_patch)
    labels.append("Min/Max range")

    ax.set_xlabel("Time steps", size=24)
    ax.set_ylabel("Number of Languages", size=24)
    ax.tick_params(axis="both", labelsize=20)
    ax.legend(
        handles=handles,
        labels=labels,
        loc="upper right",
        fontsize=16,
    )

    fig.tight_layout()

    fig.savefig(output_dir / "classification_comparison.pdf", dpi=500, bbox_inches="tight")
    plt.close()
    print("Saved classification_comparison.pdf")


def summarize_classification_methods(
    baseline_path: Path,
    classification_files: list[tuple[str, Path]],
    output_dir: Path,
) -> None:
    """Write a CSV comparing final-step metrics across classification methods.

    Wide format: one row per metric, one column per classification method (+ baseline).
    """
    all_methods: list[tuple[str, pd.DataFrame]] = [("Baseline", pd.read_csv(baseline_path))]
    for name, path in classification_files:
        all_methods.append((name, pd.read_csv(path)))

    wide_rows = []
    for metric in BASELINE_METRICS:
        row: dict = {"metric": metric, "label": METRIC_LABELS.get(metric, metric)}
        for method_name, df in all_methods:
            final_year = df["year"].max()
            df_final = df[df["year"] == final_year]
            if metric not in df_final.columns:
                row[method_name] = None
                continue
            values = df_final[metric].dropna()
            row[f"{method_name}_median"] = values.median()
            row[f"{method_name}_min"] = values.min()
            row[f"{method_name}_max"] = values.max()
        wide_rows.append(row)

    wide_df = pd.DataFrame(wide_rows)
    out_path = output_dir / "classification_summary.csv"
    wide_df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")


def create_classification_plots(baseline_path: Path, classification_files: list[tuple[str, Path]]) -> None:
    """Entry point for classification comparison mode."""
    output_dir = baseline_path.parent / "classification_plots"
    output_dir.mkdir(exist_ok=True)
    plot_classification_comparison(baseline_path, classification_files, output_dir)
    summarize_classification_methods(baseline_path, classification_files, output_dir)


# ---------------------------------------------------------------------------
# Sensitivity-only plots
# ---------------------------------------------------------------------------


def plot_state_variable(
    df: pd.DataFrame,
    variable: str,
    state_variable: str,
    baseline_values: dict,
    output_dir: Path,
    parameter_labels: dict,
) -> None:
    """Plot the language richness for each sensitivity parameter across all values."""
    agg = (
        df.groupby(variable)[state_variable]
        .agg(median_val="median", min_val="min", max_val="max")
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    base_plot(
        ax,
        agg,
        variable,
        baseline_values[variable],
        f"Average Number of {state_variable}",
        variable,
        parameter_labels,
    )
    fig.tight_layout()
    save_fig(fig, f"{state_variable}_{variable}.pdf", output_dir)


def plot_change_proportion(
    df: pd.DataFrame,
    variable: str,
    baseline_values: dict,
    output_dir: Path,
    parameter_labels: dict,
) -> None:
    """Plot the contribution of inter-communal changes for each sensitivity parameter across all values."""
    df = df.copy()
    df["change_prop"] = df["external_change"] / (df["external_change"] + df["internal_change"])
    agg = (
        df.groupby(variable)["change_prop"]
        .agg(median_val="median", min_val="min", max_val="max")
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    base_plot(
        ax,
        agg,
        variable,
        baseline_values[variable],
        "Median proportion of change due to diffusion",
        variable,
        parameter_labels,
    )
    fig.tight_layout()
    save_fig(fig, f"change_{variable}.pdf", output_dir)


def plot_speakers(
    df: pd.DataFrame,
    variable: str,
    baseline_values: dict,
    output_dir: Path,
    parameter_labels: dict,
) -> None:
    """Plot the median number of speakers per language for each sensitivity parameter across all values."""
    agg = (
        df.groupby(variable)
        .agg(
            median_val=("speaker_median", "median"),
            min_val=("speaker_min", "min"),
            max_val=("speaker_max", "max"),
        )
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    base_plot(
        ax,
        agg,
        variable,
        baseline_values[variable],
        "Number of speakers per language",
        variable,
        parameter_labels,
    )
    fig.tight_layout()
    save_fig(fig, f"speaker_{variable}.pdf", output_dir)


def plot_spider_combo(
    df: pd.DataFrame,
    baseline_values: dict,
    output_dir: Path,
    parameter_labels: dict,
    parameter_names: list[str],
) -> None:
    """Create a spider plot for parameters of the sensitivity analysis, including confidence intervals."""
    fig, ax = plt.subplots(figsize=(12, 8))
    tab10 = plt.colormaps.get_cmap("tab10")
    colors = tab10.colors[: len(parameter_names)]

    baseline_mask = pd.Series([True] * len(df), index=df.index)
    for variable, val in baseline_values.items():
        baseline_mask &= df[variable] == val
    baseline_y = df[baseline_mask]["language_count"].median()

    for i, variable in enumerate(parameter_names):
        agg = (
            df.groupby(variable)["language_count"]
            .agg(median_val="median", min_val="min", max_val="max")
            .reset_index()
        )
        agg.loc[agg[variable] == baseline_values[variable], "median_val"] = baseline_y
        agg.loc[agg[variable] == baseline_values[variable], "min_val"] = baseline_y
        agg.loc[agg[variable] == baseline_values[variable], "max_val"] = baseline_y
        agg["x_norm"] = agg[variable] / baseline_values[variable]

        label = parameter_labels.get(variable, variable)
        color = colors[i]
        ax.plot(agg["x_norm"], agg["median_val"], label=label, color=color)
        ax.fill_between(agg["x_norm"], agg["min_val"], agg["max_val"], alpha=0.2, color=color)

    handles, labels = ax.get_legend_handles_labels()
    ci_patch = Patch(facecolor="gray", alpha=0.4, label="Min/Max range")
    handles.append(ci_patch)
    labels.append("Min/Max range")

    ax.set_xlabel("Parameter Multiplication Factor (1 = baseline)", size=26)
    ax.set_ylabel("Number of Languages", size=26)
    ax.tick_params(axis="both", labelsize=22)
    ax.legend(
        handles=handles,
        labels=labels,
        loc="upper right",
        fontsize=17,
    )
    fig.tight_layout()
    save_fig(fig, "Spider.pdf", output_dir)


def plot_parameter_effects(
    df: pd.DataFrame,
    baseline_values: dict,
    output_dir: Path,
    parameter_names: list[str],
    parameter_labels: dict[str, str],
) -> None:
    """For each parameter, plot language count over time, averaged across seeds.

    One line per unique parameter value (other params held at baseline).
    """
    fig, axes = plt.subplots(len(parameter_names), 1, figsize=(10, 4 * len(parameter_names)), sharex=True)

    if len(parameter_names) == 1:
        axes = [axes]

    baseline_mask = pd.Series([True] * len(df), index=df.index)
    for variable, val in baseline_values.items():
        baseline_mask &= df[variable] == val

    baseline_df = (
        df[baseline_mask]
        .groupby("step")
        .agg(
            language_count_median=("language_count", "median"),
            language_count_min=("language_count", "min"),
            language_count_max=("language_count", "max"),
        )
        .reset_index()
    )

    for ax, param in zip(axes, parameter_names, strict=True):
        label = parameter_labels.get(param, param)
        unique_values = sorted(df[param].unique())
        colors = plt.get_cmap("viridis")(np.linspace(0, 1, len(unique_values)))

        ax.plot(
            baseline_df["step"],
            baseline_df["language_count_median"],
            label="Baseline",
            color="black",
            linestyle="--",
            linewidth=2,
        )
        ax.fill_between(
            baseline_df["step"],
            baseline_df["language_count_min"],
            baseline_df["language_count_max"],
            alpha=0.1,
            color="black",
            hatch="//",
        )

        for val, color in zip(unique_values, colors, strict=True):
            if val == baseline_values[param]:
                continue
            grouped = (
                df[df[param] == val]
                .groupby("step")
                .agg(
                    language_count_median=("language_count", "median"),
                    language_count_min=("language_count", "min"),
                    language_count_max=("language_count", "max"),
                )
                .reset_index()
            )
            ax.plot(grouped["step"], grouped["language_count_median"], label=str(val), color=color)
            ax.fill_between(
                grouped["step"],
                grouped["language_count_min"],
                grouped["language_count_max"],
                alpha=0.2,
                color=color,
            )

        ax.set_title(label)
        ax.set_ylabel("Number of Languages")
        ax.legend(title=label, fontsize=7)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time steps")
    plt.tight_layout()
    fig.savefig(output_dir / "parameter_effects.pdf", dpi=150)
    plt.close()
    print("Saved parameter_effects.pdf")


def create_sensitivity_plots(input_path: Path) -> None:
    """Entry point for sensitivity mode."""
    parameter_names = ["mutation_rate", "diffusion_rate", "radius", "similarity_preference", "speed"]

    parameter_labels = {
        "speed": "Migration speed",
        "mutation_rate": "Intra-communal change probability",
        "radius": "Diffusion radius",
        "diffusion_rate": "Diffusion probability",
        "similarity_preference": "Similarity preference",
    }

    plot_parameter_order = [
        "mutation_rate",
        "diffusion_rate",
        "radius",
        "similarity_preference",
        "speed",
    ]  # Can be removed

    baseline_values = {
        "speed": 10,
        "mutation_rate": 0.00005,
        "radius": 50.0,
        "diffusion_rate": 0.14,
        "similarity_preference": 0.6,
    }

    output_dir = input_path.parent / "sensitivity_plots"
    output_dir.mkdir(exist_ok=True)

    df_og = pd.read_csv(input_path)

    # Only select the final time step
    final_year = df_og["year"].max()
    df = df_og[df_og["year"].isin([final_year])].copy()

    for variable in parameter_names:
        other_params = [p for p in parameter_names if p != variable]
        mask = pd.Series(True, index=df.index, dtype=bool)
        for other in other_params:
            mask &= np.isclose(df[other], baseline_values[other])
        df_var = df[mask].copy()

        if df_var.empty:
            print(f"No rows found for variable '{variable}', skipping.")
            continue

        plot_state_variable(df_var, variable, "language_count", baseline_values, output_dir, parameter_labels)
        plot_state_variable(df_var, variable, "divergences", baseline_values, output_dir, parameter_labels)
        plot_state_variable(df_var, variable, "convergences", baseline_values, output_dir, parameter_labels)
        plot_change_proportion(df_var, variable, baseline_values, output_dir, parameter_labels)
        plot_speakers(df_var, variable, baseline_values, output_dir, parameter_labels)

    plot_spider_combo(df, baseline_values, output_dir, parameter_labels, plot_parameter_order)
    plot_seed_variance_base(df_og, output_dir, baseline_values=baseline_values)
    plot_parameter_effects(df_og, baseline_values, output_dir, plot_parameter_order, parameter_labels)
    plot_speaker_dist_base(df, output_dir, baseline_values=baseline_values)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create sensitivity/baseline plots of statistics file generated by spawn_summarized.py.",
    )
    parser.add_argument(
        "input_file",
        help="CSV file that contains summary statistics on different runs.",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        default=False,
        help="Set if the CSV is from a baseline run (no parameter columns).",
    )

    def named_path(value: str) -> tuple[str, Path]:
        """Parse 'name:path' into (name, Path)."""
        if ":" not in value:
            raise argparse.ArgumentTypeError(f"Expected format 'name:path', got '{value}'")
        name, path = value.split(":", maxsplit=1)
        return name.strip(), Path(path.strip())

    parser.add_argument(
        "--classification",
        nargs="+",
        type=named_path,
        default=None,
        metavar="NAME:PATH",
        help="Named CSV files for classification method comparison, e.g. 'single:path/to/single.csv'.",
    )
    args = parser.parse_args()

    if args.baseline:
        create_baseline_plots(Path(args.input_file))
    elif args.classification is not None:
        create_classification_plots(Path(args.input_file), args.classification)
    else:
        create_sensitivity_plots(Path(args.input_file))
