import pandas as pd
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.patches import Patch
from pathlib import Path
from scipy import stats

def save_fig(fig: plt.Figure, name: str, output_dir: Path) -> None:
    fig.savefig(output_dir / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {name}")


def base_plot(ax: plt.Axes, agg: pd.DataFrame, x_col: str, baseline_x: float, ylabel: str, variable: str, parameter_labels: dict) -> None:
    """Shared scatter plot logic: mean (circle) + min/max (triangles) + baseline vline."""
    ax.axvline(baseline_x, color="darkgrey", linestyle="--", linewidth=1.3)
    ax.scatter(agg[x_col], agg["mean_val"], s=80, color="black",   zorder=3, label="Mean")
    ax.scatter(agg[x_col], agg["min_val"],  s=80, color="#01204E", zorder=3, marker="^", label="Min")
    ax.scatter(agg[x_col], agg["max_val"],  s=80, color="#98C1D9", zorder=3, marker="^", label="Max")
    ax.set_xlabel(parameter_labels.get(variable, variable), fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.tick_params(axis="both", labelsize=18)
    ax.legend(fontsize=13)
    ax.grid(True, alpha=0.3)


# ── Per-variable plots ────────────────────────────────────────────────────────
def plot_language_count(df: pd.DataFrame, variable: str, baseline_values: dict, output_dir: Path, parameter_labels: dict) -> None:
    agg = (
        df.groupby(variable)["language_count"]
        .agg(mean_val="mean", min_val="min", max_val="max")
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    base_plot(ax, agg, variable, baseline_values[variable], "Average Number of Languages", variable, parameter_labels)
    #ax.set_ylim(*YLIM_LANG)
    fig.tight_layout()
    save_fig(fig, f"morris_lang_{variable}.png", output_dir)


def plot_language_count_combo(df: pd.DataFrame, baseline_values: dict, output_dir: Path, parameter_labels: dict) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    variables = list(baseline_values.keys())
    colors = plt.cm.tab10.colors

    # Compute the single baseline run result once
    baseline_mask = pd.Series([True] * len(df), index=df.index)
    for variable, val in baseline_values.items():
        baseline_mask &= (df[variable] == val)
    baseline_y = df[baseline_mask]["language_count"].mean()

    for i, variable in enumerate(variables):
        agg = (
            df.groupby(variable)["language_count"]
            .agg(mean_val="mean", min_val="min", max_val="max")
            .reset_index()
        )

        # Replace the mean at the baseline value with the true baseline run result
        agg.loc[agg[variable] == baseline_values[variable], "mean_val"] = baseline_y

        x_min = agg[variable].min()
        x_max = agg[variable].max()
        agg["x_norm"] = (agg[variable] - x_min) / (x_max - x_min)
        baseline_norm = (baseline_values[variable] - x_min) / (x_max - x_min)

        label = parameter_labels.get(variable, variable)
        color = colors[i % len(colors)]

        ax.plot(agg["x_norm"], agg["mean_val"], label=label, color=color)
        ax.axvline(x=baseline_norm, color=color, linestyle="--", linewidth=0.8, alpha=0.6)

    ax.set_xlabel("Normalized Parameter Value (0 = min, 1 = max)")
    ax.set_ylabel("Average Number of Languages")
    ax.legend(title="Parameter", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_xlim(0, 1)

    fig.tight_layout()
    save_fig(fig, "lang_combo.png", output_dir)


def plot_spider_combo(df: pd.DataFrame, baseline_values: dict, output_dir: Path, parameter_labels: dict) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    variables = list(baseline_values.keys())
    colors = plt.cm.tab10.colors

    # Compute the single baseline run result once
    baseline_mask = pd.Series([True] * len(df), index=df.index)
    for variable, val in baseline_values.items():
        baseline_mask &= (df[variable] == val)
    baseline_y = df[baseline_mask]["language_count"].mean()
    print(baseline_y)

    for i, variable in enumerate(variables):
        agg = (
            df.groupby(variable)["language_count"]
            .agg(mean_val="mean")
            .reset_index()
        )

        # Replace the mean at the baseline value with the true baseline run result
        agg.loc[agg[variable] == baseline_values[variable], "mean_val"] = baseline_y

        # Normalize x to multiplication factor relative to baseline
        baseline_val = baseline_values[variable]
        agg["x_norm"] = agg[variable] / baseline_val

        # Normalize y relative to baseline result
        agg["y_norm"] = agg["mean_val"] / baseline_y

        label = parameter_labels.get(variable, variable)
        color = colors[i % len(colors)]

        #ax.plot(agg["x_norm"], agg["y_norm"], label=label, color=color)
        ax.plot(agg["x_norm"], agg["mean_val"], label=label, color=color)

    # Mark the baseline point where all lines converge
    #ax.scatter([1], [1], color="black", zorder=5, s=80, label="Baseline")
    #ax.axvline(x=1, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    #ax.axhline(y=1, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)

    ax.set_xlabel("Multiplication Factor (1 = baseline)", size = 20)
    ax.set_ylabel("Number of Languages", size = 20)
    ax.tick_params(axis="both", labelsize=16)
    ax.legend(title="Parameter", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_title("Sensitivity analysis result", size=20)

    fig.tight_layout()
    save_fig(fig, "lang_spider.png", output_dir)


def plot_spider_combo_ci(df: pd.DataFrame, baseline_values: dict, output_dir: Path, parameter_labels: dict) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    variables = list(baseline_values.keys())
    colors = plt.cm.tab10.colors

    # Compute the single baseline run result once
    baseline_mask = pd.Series([True] * len(df), index=df.index)
    for variable, val in baseline_values.items():
        baseline_mask &= (df[variable] == val)
    baseline_y = df[baseline_mask]["language_count"].mean()
    print(baseline_y)

    for i, variable in enumerate(variables):
        # Group and aggregate with confidence intervals
        agg = (
            df.groupby(variable)["language_count"]
            .agg(
                mean_val="mean",
                std_val="std",
                count="count"
            )
            .reset_index()
        )

        # Calculate 95% confidence interval
        # CI = mean ± t * (std / sqrt(n))
        agg["sem"] = agg["std_val"] / np.sqrt(agg["count"])  # Standard error of mean
        agg["ci_lower"] = agg["mean_val"] - stats.t.ppf(0.975, agg["count"]-1) * agg["sem"]
        agg["ci_upper"] = agg["mean_val"] + stats.t.ppf(0.975, agg["count"]-1) * agg["sem"]

        # Replace the mean at the baseline value with the true baseline run result
        agg.loc[agg[variable] == baseline_values[variable], "mean_val"] = baseline_y
        agg.loc[agg[variable] == baseline_values[variable], "ci_lower"] = baseline_y
        agg.loc[agg[variable] == baseline_values[variable], "ci_upper"] = baseline_y

        # Normalize x to multiplication factor relative to baseline
        baseline_val = baseline_values[variable]
        agg["x_norm"] = agg[variable] / baseline_val

        # Normalize y relative to baseline result (if needed)
        agg["y_norm"] = agg["mean_val"] / baseline_y

        label = parameter_labels.get(variable, variable)
        color = colors[i % len(colors)]

        # Plot the line
        ax.plot(agg["x_norm"], agg["mean_val"], label=label, color=color)

        # Add confidence interval as shaded region
        ax.fill_between(
            agg["x_norm"],
            agg["ci_lower"],
            agg["ci_upper"],
            alpha=0.2,
            color=color,
            label= None  # Only label once
        )

    # Add general variable for CI in the legend
    handles, labels = ax.get_legend_handles_labels()
    ci_patch = Patch(facecolor="gray", alpha=0.4, label="95% CI")
    handles.append(ci_patch)
    labels.append("95% CI")

    ax.legend(handles=handles, labels=labels, title="Parameter", bbox_to_anchor=(1.05, 1), loc="upper left")

    ax.set_xlabel("Parameter Multiplication Factor (1 = baseline)", size = 20)
    ax.set_ylabel("Number of Languages", size = 20)
    ax.tick_params(axis="both", labelsize=16)
    #ax.legend(title="Parameter", bbox_to_anchor=(1.05, 1), loc="upper left")
    #ax.set_title("Sensitivity analysis of community-level processes", size=20)

    fig.tight_layout()
    save_fig(fig, "lang_spider.pdf", output_dir)


def plot_change_proportion(df: pd.DataFrame, variable: str, baseline_values: dict, output_dir: Path, parameter_labels: dict) -> None:
    df = df.copy()
    df["change_prop"] = df["external_change"] / (df["external_change"] + df["internal_change"])

    agg = (
        df.groupby(variable)["change_prop"]
        .agg(mean_val="mean", min_val="min", max_val="max")
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    base_plot(ax, agg, variable, baseline_values[variable], "Average proportion of change due to diffusion", variable, parameter_labels)
    #ax.set_ylim(*YLIM_CHANGE)
    fig.tight_layout()
    save_fig(fig, f"morris_change_{variable}.png", output_dir)


def plot_speakers(df: pd.DataFrame, variable: str, baseline_values: dict, output_dir: Path, parameter_labels: dict) -> None:
    agg = (
        df.groupby(variable)
        .agg(
            mean_val=("speaker_mean", "mean"),
            min_val=("speaker_min",  "min"),
            max_val=("speaker_max",  "max"),
        )
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    base_plot(ax, agg, variable, baseline_values[variable], "Number of speakers per language", variable, parameter_labels)
    #ax.set_ylim(*YLIM_SPEAKERS)
    fig.tight_layout()
    save_fig(fig, f"morris_speaker_{variable}.png", output_dir)


def plot_speaker_distribution_base(df: pd.DataFrame, baseline_values: dict, output_dir: Path, bin_step: int = 1) -> None:
    """Plot speaker distribution for the baseline parameter combination for the last time step per seed."""

    # Filter to baseline run
    baseline_mask = pd.Series([True] * len(df), index=df.index)
    for variable, val in baseline_values.items():
        baseline_mask &= (df[variable] == val)
    df_baseline = df[baseline_mask].copy()

    # Get only the last time step
    last_step = df_baseline["Year"].max()
    df_last = df_baseline[df_baseline["Year"] == last_step & df_baseline["speakers"].notna()]

    seeds = sorted(df_last["seed"].unique())

    for seed in seeds:
        df_seed = df_last[df_last["seed"] == seed]
        all_speakers = np.concatenate(df_seed["speakers"].values)

        bins = np.arange(0, all_speakers.max() + bin_step, bin_step)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        hist, _ = np.histogram(all_speakers, bins=bins)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(bin_centers, hist, width=np.diff(bins), alpha=0.6, color="steelblue")

        param_str = ", ".join(f"{k}={v}" for k, v in baseline_values.items())
        ax.set_title(f"Speaker Distribution (last time step, seed {seed})\n{param_str}", size=14)
        ax.set_xlabel("Number of Agents per Language", size=14)
        ax.set_xlim(0, 80)
        ax.set_ylabel("Frequency", size=14)
        ax.tick_params(axis="both", labelsize=11)
        fig.tight_layout()
        save_fig(fig, f"speaker_dist_baseline_seed{seed}.png", output_dir)
        plt.close(fig)


def plot_seed_variance_base(df: pd.DataFrame, baseline_values: dict, output_path: Path) -> None:
    """Plot language count per seed for each parameter combination to check variance."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Filter to baseline run
    baseline_mask = pd.Series([True] * len(df), index=df.index)
    for variable, val in baseline_values.items():
        baseline_mask &= (df[variable] == val)
    df_baseline = df[baseline_mask].copy()

    # Get unique seeds and assign colors
    seeds = sorted(df_baseline["seed"].unique())
    colors = cm.viridis(np.linspace(0, 1, len(seeds)))

    # Plot each seed's trajectory
    for seed, color in zip(seeds, colors):
        seed_df = df_baseline[df_baseline["seed"] == seed]
        grouped = seed_df.groupby("year")["language_count"].mean()
        ax.plot(grouped.index, grouped.values, color=color, alpha=0.7, linewidth=1, label=f'Seed {seed}')

    # Create a formatter that divides the value by 1000
    def format_kyears(x, pos):
        return f'{int(x / 1000)}'

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_kyears))
    ax.set_title("Number of languages over time for the baseline parameter set", size=20)
    ax.set_xlabel("Time (k years)", size = 20)
    ax.set_ylabel("Language Count", size= 20)
    ax.tick_params(axis='both', labelsize=16)
    #ax.legend(bbox_to_anchor=(-1.05, 1), loc='upper left')
    ax.legend(loc='upper right', fontsize = "xx-large")

    ax.xaxis.set_major_locator(ticker.MultipleLocator(base=20000))

    plt.tight_layout()

    fig.savefig(output_path / "seed_variance_base.pdf", dpi=500, bbox_inches='tight')
    plt.close()
    print(f"Saved to {output_path / 'seed_variance_base.png'}")


def plot_parameter_effects(df: pd.DataFrame, baseline_values: dict, output_path: Path, parameter_names: list[str], parameter_labels: dict) -> None:
    """For each parameter, plot language count over time, averaged across seeds,
    with one line per unique parameter value (other params held at their median)."""

    fig, axes = plt.subplots(len(parameter_names), 2, figsize=(14, 4 * len(parameter_names)))

    # Compute the single baseline run result once
    baseline_mask = pd.Series([True] * len(df), index=df.index)
    for variable, val in baseline_values.items():
        baseline_mask &= (df[variable] == val)

    # Get baseline data for plotting
    baseline_df = df[baseline_mask].groupby("year").agg(
        language_count_mean=("language_count", "mean"),
        language_count_std=("language_count", "std"),
        speaker_mean_mean=("speaker_mean", "mean"),
        speaker_mean_std=("speaker_mean", "std"),
    ).reset_index()

    baseline_y = baseline_df["language_count_mean"].mean()  # Overall baseline average

    for row_idx, param in enumerate(parameter_names):
        ax_lang = axes[row_idx, 0]
        ax_speak = axes[row_idx, 1]

        unique_values = sorted(df[param].unique())
        colors = cm.viridis(np.linspace(0, 1, len(unique_values)))

        # First, plot the baseline line (same across all subplots)
        ax_lang.plot(baseline_df["year"], baseline_df["language_count_mean"],
                     label="Baseline", color="black", linestyle="--", linewidth=2)
        ax_lang.fill_between(
            baseline_df["year"],
            baseline_df["language_count_mean"] - baseline_df["language_count_std"],
            baseline_df["language_count_mean"] + baseline_df["language_count_std"],
            alpha=0.1, color="black", hatch="//"
        )

        ax_speak.plot(baseline_df["year"], baseline_df["speaker_mean_mean"],
                      label="Baseline", color="black", linestyle="--", linewidth=2)
        ax_speak.fill_between(
            baseline_df["year"],
            baseline_df["speaker_mean_mean"] - baseline_df["speaker_mean_std"],
            baseline_df["speaker_mean_mean"] + baseline_df["speaker_mean_std"],
            alpha=0.1, color="black", hatch="//"
        )

        # Then plot parameter variations
        for val, color in zip(unique_values, colors):
            subset = df[df[param] == val]

            # Skip if this IS the baseline value for this parameter
            if val == baseline_values[param]:
                continue  # Already plotted above

            grouped = subset.groupby("year").agg(
                language_count_mean=("language_count", "mean"),
                language_count_std=("language_count", "std"),
                speaker_mean_mean=("speaker_mean", "mean"),
                speaker_mean_std=("speaker_mean", "std"),
            ).reset_index()

            # Language count
            ax_lang.plot(grouped["year"], grouped["language_count_mean"],
                        label=str(val), color=color)
            ax_lang.fill_between(
                grouped["year"],
                grouped["language_count_mean"] - grouped["language_count_std"],
                grouped["language_count_mean"] + grouped["language_count_std"],
                alpha=0.2, color=color,
            )

            # Speaker mean
            ax_speak.plot(grouped["year"], grouped["speaker_mean_mean"],
                         label=str(val), color=color)
            ax_speak.fill_between(
                grouped["year"],
                grouped["speaker_mean_mean"] - grouped["speaker_mean_std"],
                grouped["speaker_mean_mean"] + grouped["speaker_mean_std"],
                alpha=0.2, color=color,
            )

            label = parameter_labels.get(val, val)

        ax_lang.set_title(f"{param} — Language Count")
        ax_lang.set_xlabel("Year")
        ax_lang.set_ylabel("Number of Languages")
        ax_lang.legend(title=param, fontsize=7)

        ax_speak.set_title(f"{param} — Mean Speakers per Language")
        ax_speak.set_xlabel("Year")
        ax_speak.set_ylabel("Mean Speakers")
        ax_speak.legend(title=param, fontsize=7)

    plt.tight_layout()
    fig.savefig(output_path / "parameter_effects.png", dpi=150)
    plt.close()
    print(f"Saved to {output_path / 'parameter_effects.png'}")


def main() -> None:
    parameter_names = ["speed", "mutation_rate", "radius", "diffusion_rate", "similarity_preference"]

    parameter_labels = {
        "speed": "Migration speed",
        "mutation_rate": "Internal change rate",
        "radius": "Radius of contact",
        "diffusion_rate": "Diffusion rate",
        "similarity_preference": "Similarity preference",
    }

    baseline_values = {
        "speed":                 20,
        "mutation_rate":         0.00005,
        "radius":                50.0,
        "diffusion_rate":        0.1,
        "similarity_preference": 0.6,
    }

    input_path = Path("C:/Users/Posma002/OneDrive - Universiteit Utrecht/Agentbased_Linguistics/Code/MinimalModel/lecoOutput/xtreme_tests_int+/veloc_sens_L3_5/spawn_combined_stats_base.csv")
    output_dir = input_path.parent

    #OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    years_to_consider = 200

    df_og = pd.read_csv(input_path)
    last = df_og["year"].max()
    df = df_og[df_og["year"] >= last - years_to_consider].copy()
    df = df[df["similarity_preference"] > -2]

    for variable in parameter_names:
        # Each parameter's rows are identified by that column having varying values
        # while the others are fixed at their baseline — filter to rows where all
        # other parameters are at baseline so we isolate one parameter at a time
        other_params = [p for p in parameter_names if p != variable]
        mask = pd.Series(True, index=df.index)
        for other in other_params:
            mask &= np.isclose(df[other], baseline_values[other])
        df_var = df[mask].copy()

        if df_var.empty:
            print(f"No rows found for variable '{variable}', skipping.")
            continue

        plot_language_count(df_var, variable, baseline_values, output_dir, parameter_labels)
        plot_change_proportion(df_var, variable, baseline_values, output_dir, parameter_labels)
        plot_speakers(df_var, variable, baseline_values, output_dir, parameter_labels)

    df = df[df["radius"] > 0.0]
    plot_language_count_combo(df, baseline_values, output_dir, parameter_labels)
    plot_spider_combo(df, baseline_values, output_dir, parameter_labels)
    plot_spider_combo_ci(df, baseline_values, output_dir, parameter_labels)
    plot_seed_variance_base(df_og, baseline_values, output_dir)
    plot_parameter_effects(df_og, baseline_values, output_dir, parameter_names, parameter_labels)
    #plot_speaker_distribution_base(df, baseline_values, output_dir)


if __name__ == "__main__":
    main()
