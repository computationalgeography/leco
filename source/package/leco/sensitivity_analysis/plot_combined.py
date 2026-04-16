import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from pathlib import Path

PARAMETER_NAMES = [
    "speed",
    "mutation_rate",
    "radius",
    "diffusion_rate",
    "similarity_preference",
]

def plot_parameter_effects(df: pd.DataFrame, output_path: Path) -> None:
    """For each parameter, plot language count over time, averaged across seeds,
    with one line per unique parameter value (other params held at their median)."""

    fig, axes = plt.subplots(len(PARAMETER_NAMES), 2, figsize=(14, 4 * len(PARAMETER_NAMES)))

    for row_idx, param in enumerate(PARAMETER_NAMES):
        ax_lang = axes[row_idx, 0]
        ax_speak = axes[row_idx, 1]

        unique_values = sorted(df[param].unique())
        colors = cm.viridis(np.linspace(0, 1, len(unique_values)))

        for val, color in zip(unique_values, colors):
            subset = df[df[param] == val]

            # Average across seeds and all other parameter combinations
            grouped = subset.groupby("year").agg(
                language_count_mean=("language_count", "mean"),
                language_count_std=("language_count", "std"),
                speaker_mean_mean=("speaker_mean", "mean"),
                speaker_mean_std=("speaker_mean", "std"),
            ).reset_index()

            # Language count
            ax_lang.plot(grouped["year"], grouped["language_count_mean"], label=str(val), color=color)
            ax_lang.fill_between(
                grouped["year"],
                grouped["language_count_mean"] - grouped["language_count_std"],
                grouped["language_count_mean"] + grouped["language_count_std"],
                alpha=0.2, color=color,
            )

            # Speaker mean
            ax_speak.plot(grouped["year"], grouped["speaker_mean_mean"], label=str(val), color=color)
            ax_speak.fill_between(
                grouped["year"],
                grouped["speaker_mean_mean"] - grouped["speaker_mean_std"],
                grouped["speaker_mean_mean"] + grouped["speaker_mean_std"],
                alpha=0.2, color=color,
            )

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


def plot_seed_variance(df: pd.DataFrame, output_path: Path) -> None:
    """Plot language count per seed for each parameter combination to check variance."""
    # Pick one parameter at a time, fix others at median
    fig, axes = plt.subplots(1, len(PARAMETER_NAMES), figsize=(5 * len(PARAMETER_NAMES), 4))

    for ax, param in zip(axes, PARAMETER_NAMES):
        unique_values = sorted(df[param].unique())
        colors = cm.viridis(np.linspace(0, 1, len(unique_values)))

        for val, color in zip(unique_values, colors):
            subset = df[df[param] == val]
            for seed, seed_df in subset.groupby("seed"):
                grouped = seed_df.groupby("year")["language_count"].mean()
                ax.plot(grouped.index, grouped.values, color=color, alpha=0.3, linewidth=0.8)

        ax.set_title(f"Seed variance\n{param}")
        ax.set_xlabel("Year")
        ax.set_ylabel("Language Count")

    plt.tight_layout()
    fig.savefig(output_path / "seed_variance.png", dpi=150)
    plt.close()
    print(f"Saved to {output_path / 'seed_variance.png'}")


def plot_speaker_distribution(df: pd.DataFrame, output_path: Path, bin_step: int = 1) -> None:
    """Plot histogram of speakers per language, averaged over seeds for the last 10 timesteps."""
    import ast

    df_last10 = df[df["speakers"].notna()].copy()

    # Parse strings back to numeric numpy arrays when loaded from CSV
    def parse_speaker_array(val):
        if isinstance(val, np.ndarray):
            return val.astype(float)
        if isinstance(val, str):
            # Handles both "[1 2 3]" (numpy style) and "[1, 2, 3]" (list style)
            cleaned = val.strip("[]").replace(",", " ").split()
            return np.array(cleaned, dtype=float)
        return np.array(val, dtype=float)

    df_last10["speakers"] = df_last10["speakers"].apply(parse_speaker_array)

    param_combos = df_last10.groupby(PARAMETER_NAMES)

    for param_values, group in param_combos:
        param_dict = dict(zip(PARAMETER_NAMES, param_values))

        seed_distributions = []
        for seed, seed_group in group.groupby("seed"):
            all_speakers = np.concatenate(seed_group["speakers"].values)
            #all_speakers = np.log(all_speakers)
            seed_distributions.append(all_speakers)

        all_values = np.concatenate(seed_distributions)
        bins = np.arange(0, all_values.max() + bin_step, bin_step)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        """ bins = np.linspace(0, all_values.max(), 30)
        bin_centers = 0.5 * (bins[:-1] + bins[1:]) """

        histograms = np.array([
            np.histogram(dist, bins=bins)[0]
            for dist in seed_distributions
        ])

        mean_hist = histograms.mean(axis=0)
        std_hist = histograms.std(axis=0)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(bin_centers, mean_hist, width=np.diff(bins), alpha=0.6, color="steelblue", label="Mean across seeds")
        #ax.errorbar(bin_centers, mean_hist, yerr=[np.zeros_like(std_hist), std_hist], fmt="none", color="steelblue", capsize=3, linewidth=1, label="±1 std across seeds")

        param_str = ", ".join(f"{k}={v}" for k, v in param_dict.items())
        ax.set_title(f"Speaker Distribution (last 10 steps)\n{param_str}", size=14)
        ax.set_xlabel("Number of Agents per Language", size=14)
        ax.set_xlim(0,80)
        ax.set_ylabel("Frequency", size=14)
        ax.tick_params(axis="both", labelsize=11)
        ax.legend()

        plt.tight_layout()
        fname = "_".join(f"{k}{v}" for k, v in param_dict.items())
        fig.savefig(output_path / f"speaker_dist_{fname}.png", dpi=150)
        plt.close()
        print(f"Saved speaker distribution for {param_str}")


def plot_speaker_distribution_step(df: pd.DataFrame, output_path: Path, bin_step: int = 1) -> None:
    """Plot histogram of speakers per language, averaged over seeds and last 10 timesteps."""

    df_last10 = df[df["speakers"].notna()].copy()

    def parse_speaker_array(val):
        if isinstance(val, np.ndarray):
            return val.astype(float)
        if isinstance(val, str):
            cleaned = val.strip("[]").replace(",", " ").split()
            return np.array(cleaned, dtype=float)
        return np.array(val, dtype=float)

    df_last10["speakers"] = df_last10["speakers"].apply(parse_speaker_array)

    param_combos = df_last10.groupby(PARAMETER_NAMES)

    for param_values, group in param_combos:
        param_dict = dict(zip(PARAMETER_NAMES, param_values))

        # Collect one array per (seed, time_step) row
        raw = [
            row["speakers"]
            for _, row in group.iterrows()
            if len(row["speakers"]) > 0
        ]

        if not raw:
            continue

        # Define bins from global max across all time steps and seeds
        all_values = np.concatenate(raw)
        bins = np.arange(0, all_values.max() + bin_step, bin_step)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])

        # One histogram per (seed, time_step) — then average
        histograms = np.array([np.histogram(speakers, bins=bins)[0] for speakers in raw])
        mean_hist = histograms.mean(axis=0)
        std_hist = histograms.std(axis=0)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(bin_centers, mean_hist, width=np.diff(bins), alpha=0.6, color="steelblue", label="Mean across seeds and time steps")
        ax.errorbar(bin_centers, mean_hist, yerr=[np.zeros_like(std_hist), std_hist], fmt="none", color="steelblue", capsize=3, linewidth=1, label="±1 std across seeds")

        param_str = ", ".join(f"{k}={v}" for k, v in param_dict.items())
        ax.set_title(f"Speaker Distribution (last 10 steps)\n{param_str}", size=14)
        ax.set_xlabel("Number of Agents per Language", size=14)
        ax.set_xlim(0, 80)
        ax.set_ylabel("Mean Frequency per Time Step", size=14)
        ax.tick_params(axis="both", labelsize=11)
        ax.legend()

        plt.tight_layout()
        fname = "_".join(f"{k}{v}" for k, v in param_dict.items())
        fig.savefig(output_path / f"distribution/speaker_dist_{fname}.png", dpi=150)
        plt.close()
        print(f"Saved speaker distribution for {param_str}")


if __name__ == "__main__":
    csv_path = Path("C:/Users/Posma002/OneDrive - Universiteit Utrecht/Agentbased_Linguistics/Code/MinimalModel/lecoOutput/xtreme_tests_int+/veloc_sens_L3_long/spawn_combined_stats.csv")
    df = pd.read_csv(csv_path)

    # Ensure correct types
    for param in PARAMETER_NAMES:
        df[param] = df[param].astype(float)

    output_path = csv_path.parent
    plot_parameter_effects(df, output_path)
    plot_seed_variance(df, output_path)
    plot_speaker_distribution_step(df, output_path)
