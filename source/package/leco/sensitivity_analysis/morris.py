"""Script to perform a Morris sensitivity analysis"""

from pathlib import Path
from SALib.sample import morris as morris_sample
from SALib.analyze import morris as morris_analyze
import logging
import numpy as np
import json

from ..model.simulation import simulate
from ..cluster.all import classify_all
from ..cluster.feed_forward import diversify


def modify_config(
    configuration: dict,
    params: list[float],
    param_names: list[str],
    seed: int,
) -> dict:
    """Modify the configuration parameters to the sensitivity run."""

    modified_config = configuration.copy()

    param_mapping = {
        "speed": ["movement", "speed"],
        "mutation_rate": ["language", "mutation_rate"],
        "radius": ["interaction", "radius"],
        "diffusion_rate": ["interaction", "diffusion_rate"],
        "similarity_preference": ["interaction", "similarity_preference"],
    }

    for name, value in zip(param_names, params):
        path = param_mapping[name]
        if len(path) == 2:
            # Nested structure: config[section][key]
            modified_config[path[0]][path[1]] = value
        else:
            # Flat structure: config[key]
            modified_config[path[0]] = value

    modified_config["initialization"]["seed"] = seed

    return modified_config


def model(
    cluster_method: str,
    distance_threshold: float,
    configuration: dict,
    directory_path: Path,
    linkage: str = "average",
) -> tuple[float, float]:
    """Run the leco model."""

    # Run simulation
    diffusion_proportion = simulate(configuration, directory_path, sensitivity=True)

    if cluster_method == "all":
        language_number = classify_all(
            directory_path,
            distance_threshold,
            linkage,
            sensitivity=True,
        )
    elif cluster_method == "feed_forward":
        # Feed-forward needs interaction radius as input
        language_number = diversify(
            directory_path,
            distance_threshold,
            configuration["interaction"]["radius"],
            linkage,
            sensitivity=True,
        )
    else:
        logging.error(f"Unknown clustering method: {cluster_method}")

    return diffusion_proportion, language_number


def run_model(
    params: list[float],
    run_index: int,
    param_names: list[str],
    seed: int,
    cluster_method: str,
    distance_threshold: float,
    base_parameters: dict,
    directory_path: Path,
    linkage: str = "average",
) -> tuple[float, float]:
    """Run the leco model with given parameters and return a scalar output for sensitivity analysis."""

    # Create a unique output directory for this run
    run_directory_path = directory_path / f"run_{run_index:04d}"
    run_directory_path.mkdir(parents=True, exist_ok=True)

    # Create a modified config file with the current parameters
    modified_config = modify_config(base_parameters, params, param_names, seed)

    # Store a copy of the configuration file for documentation and reproducibility
    modified_config_file_path = run_directory_path / "configuration.json"
    with open(modified_config_file_path, "w") as f:
        json.dump(modified_config, f, indent=2)

    # Run the model
    try:
        logging.info(f"Running model with config: {modified_config}")
        diffusion_prob, language_number = model(
            cluster_method,
            distance_threshold,
            modified_config,
            run_directory_path,
            linkage,
        )
        logging.info(
            f"diffusion_prob: {diffusion_prob}, language_number: {language_number}"
        )
        # Convert to float for compatibility with the sensitivity analysis
        return float(diffusion_prob), float(language_number)

    except Exception as e:
        logging.error(f"Model run failed with parameters: {params}")
        logging.error(f"Error: {str(e)}")
        raise


def handle_results(
    results: dict, problem: dict, directory_path: Path, analysis_name: str
) -> None:
    """Handle and save the sensitivity analysis results."""
    results_combined = {
        "mu_star": results["mu_star"].tolist(),
        "mu": results["mu"].tolist(),
        "sigma": results["sigma"].tolist(),
        "mu_star_conf": results["mu_star_conf"].tolist(),
        "names": problem["names"],
    }

    output_file = directory_path / f"morris_results_{analysis_name}.json"
    with open(output_file, "w") as f:
        json.dump(results_combined, f, indent=2)

    print(f"SENSITIVITY ANALYSIS {analysis_name} SUMMARY")
    for i, name in enumerate(problem["names"]):
        print(f"\n{name}:")
        print(f"  μ* (importance): {results['mu_star'][i]:.4f}")
        print(f"  σ (interactions): {results['sigma'][i]:.4f}")
        print(f"  μ (direction): {results['mu'][i]:.4f}")


def analyze(
    cluster_method: str,
    distance_threshold: float,
    base_parameters: dict,
    directory_path: Path,
) -> None:
    """Run sensitivity analysis on the leco model."""

    # Morris sampling parameters
    r = 30
    num_levels = 10

    # Set the parameter value bounds
    bounds = [
        [0, 500],  # speed bounds
        [0.0, 1.0],  # mutation_rate bounds
        [0.0, 500.0],  # radius bounds
        [0.0, 1.0],  # diffusion_rate bounds
        [0.0, 1.0],  # similarity_preference bounds
    ]

    param_names = [
        "speed",
        "mutation_rate",
        "radius",
        "diffusion_rate",
        "similarity_preference",
    ]

    problem = {
        "num_vars": len(param_names),
        "names": param_names,
        "bounds": bounds,
    }

    # Generate Morris sample with r trajectories
    param_values = morris_sample.sample(
        problem, N=r, num_levels=num_levels
    )  # , grid_jump=grid_jump)

    """ param_values = [
        [10.0, 0.001, 40.0, 0.001, 0.1],
        [50.0, 0.01, 50.0, 0.01, 0.3],
        [100.0, 0.1, 60.0, 0.1, 0.4],
        [60.0, 0.5, 100.0, 0.5, 0.5],
        [200.0, 0.8, 200.0, 0.8, 0.4],
        [300.0, 1.0, 250.0, 1.0, 0.6],
    ]"""

    # param_values = [[10.0, 0.001, 40.0, 0.001, 0.1], [50.0, 0.01, 50.0, 0.01, 0.3]]

    seeds = [42]
    linkage = "average"

    # logging.debug(f"Total model runs required: {len(param_values)}")

    for seed in seeds:
        diffusion_proportions = []
        language_number = []

        run_directory_path = directory_path / f"holythree_{seed}"
        run_directory_path.mkdir(parents=True, exist_ok=True)
        for i, params in enumerate(param_values, 1):
            logging.debug(f"\nRun {i}/{len(param_values)}")
            logging.debug(f"  Parameters: {dict(zip(problem['names'], params))}")
            diffusion_prop, language_nr = run_model(
                params,
                i,
                problem["names"],
                seed,
                cluster_method,
                distance_threshold,
                base_parameters,
                run_directory_path,
                linkage,
            )
            diffusion_proportions.append(diffusion_prop)
            language_number.append(language_nr)

        diffusion_proportions = np.array(diffusion_proportions)
        language_number = np.array(language_number)

        np.savetxt(
            run_directory_path / "sensitivity_analysis_diff.csv",
            diffusion_proportions,
            delimiter=",",
        )

        np.savetxt(
            run_directory_path / "sensitivity_analysis_lang.csv",
            language_number,
            delimiter=",",
        )

    # Analyze
    diffusion_analysis = morris_analyze.analyze(
        problem,
        param_values,
        diffusion_proportions,
        conf_level=0.95,
        print_to_console=True,
        num_levels=num_levels,
    )

    number_analysis = morris_analyze.analyze(
        problem,
        param_values,
        language_number,
        conf_level=0.95,
        print_to_console=True,
        num_levels=num_levels,
    )

    handle_results(number_analysis, problem, directory_path, "language_number")

    handle_results(diffusion_analysis, problem, directory_path, "diffusion")
