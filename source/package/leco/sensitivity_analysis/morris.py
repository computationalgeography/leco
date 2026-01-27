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
    linkage: str,
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
    sens_name: str,
    param_index: int,
    # run_index: int,
    param_names: list[str],
    seed: int,
    cluster_method: str,
    distance_threshold: float,
    base_parameters: dict,
    directory_path: Path,
    linkage: str,
) -> tuple[float, float]:
    """Run the leco model with given parameters and return a scalar output for sensitivity analysis."""

    # Create a unique output directory for this run
    # run_directory_path = directory_path / f"run_{run_index:04d}"
    run_directory_path = directory_path / f"{sens_name}_{params[param_index]}"
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
        logging.info(f"diffusion_prob: {diffusion_prob}, language_number: {language_number}")
        # Convert to float for compatibility with the sensitivity analysis
        return float(diffusion_prob), float(language_number)

    except Exception as e:
        logging.error(f"Model run failed with parameters: {sens_name}")
        logging.error(f"Error: {str(e)}")
        raise


def handle_results(results: dict, problem: dict, directory_path: Path, analysis_name: str) -> None:
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


def run_per_seed(
    seeds: list[int],
    directory_path: Path,
    ofat: list[float | int],
    baseline_values: list[float],
    ofat_interest: str,
    problem: dict,
    cluster_method: str,
    distance_threshold: float,
    base_parameters: dict,
    linkage: str,
) -> tuple[np.ndarray]:
    # Get the index of the parameter being varied
    param_index = problem["names"].index(ofat_interest)

    for seed in seeds:
        diffusion_proportions = []
        language_number = []

        # If running multiple seeds, create a path for every seed
        if len(seeds) > 1:
            run_directory_path = directory_path / f"holythree_{seed}"
            run_directory_path.mkdir(parents=True, exist_ok=True)
        else:
            run_directory_path = directory_path

        # for i, params in enumerate(param_values, 1):
        for value in ofat:
            # logging.debug(f"\nRun {i}/{len(param_values)}")
            # logging.debug(f"  Parameters: {dict(zip(problem['names'], params))}")
            baseline_values[param_index] = value
            diffusion_prop, language_nr = run_model(
                baseline_values,
                ofat_interest,
                param_index,
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

    return diffusion_proportions, language_number


def create_morris_parameters(
    sensitivity_problem: dict, parameter_names: list[str], directory_path: Path, r, num_levels
) -> np.ndarray[float]:
    """Generate parameter combinations following Morris global sensitivity approach"""

    # Generate Morris sample with r trajectories
    param_values = morris_sample.sample(
        sensitivity_problem, N=r, num_levels=num_levels
    )  # , grid_jump=grid_jump)
    type(param_values)
    np.savetxt(
        directory_path / "sensitivity_parameter_values.csv",
        param_values,
        delimiter=",",
        header=parameter_names,
    )

    return param_values


def analyze(
    cluster_method: str,
    distance_threshold: float,
    base_parameters: dict,
    directory_path: Path,
) -> None:
    """Run sensitivity analysis on the leco model."""

    morris = False
    # Morris sampling parameters
    r = 20
    num_levels = 8

    # Set the parameter value bounds
    bounds = [
        [0, 100],  # speed bounds
        [0.0, 0.5],  # mutation_rate bounds
        [0.0, 100.0],  # radius bounds
        [0.0, 1.0],  # diffusion_rate bounds
        [0.0, 0.5],  # similarity_preference bounds
    ]

    parameter_names = [
        "speed",
        "mutation_rate",
        "radius",
        "diffusion_rate",
        "similarity_preference",
    ]

    sensitivity_problem = {
        "num_vars": len(parameter_names),
        "names": parameter_names,
        "bounds": bounds,
    }

    if morris:
        parameter_values = create_morris_parameters(
            sensitivity_problem, parameter_names, directory_path, r, num_levels
        )
    else:
        # parameter_values = [50.0, 0.1, 50.0, 0.1, 0.3]
        # parameter_values = [50.0, 0.05, 50.0, 0.05, 0.5]
        parameter_values = [20.0, 0.1, 50.0, 0.1, 0.5]

    # Parameter combination values for initial runs 1 to 6
    """ param_values = [
        [10.0, 0.001, 40.0, 0.001, 0.1],
        [50.0, 0.01, 50.0, 0.01, 0.3],
        [100.0, 0.1, 60.0, 0.1, 0.4],
        [60.0, 0.5, 100.0, 0.5, 0.5],
        [200.0, 0.8, 200.0, 0.8, 0.4],
        [300.0, 1.0, 250.0, 1.0, 0.6],
    ]"""

    # Skip 0.1
    ofat_rates = [0.0, 0.05, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

    ofat_distances = [0, 10, 20, 30, 40, 60, 70, 80, 90, 100]
    ofat_distances2 = [0, 10, 30, 40, 50, 60, 70, 80, 90, 100]
    ofat_sim = [0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0]

    ofat = {
        "speed": ofat_distances2,
        "mutation_rate": ofat_rates,
        "radius": ofat_distances,
        "diffusion_rate": ofat_rates,
        "similarity_preference": ofat_sim,
    }

    seeds = [
        42,
        43,
        44,
        45,
        46,
    ]
    linkage = "average"

    # logging.debug(f"Total model runs required: {len(param_values)}")

    for parameter, value_range in ofat.items():
        baseline = parameter_values.copy()

        run_directory_path = directory_path / parameter
        run_directory_path.mkdir(parents=True, exist_ok=True)

        diffusion_proportions, language_number = run_per_seed(
            seeds,
            run_directory_path,
            value_range,
            baseline,
            parameter,
            sensitivity_problem,
            cluster_method,
            distance_threshold,
            base_parameters,
            linkage,
        )

    if morris:
        # Analyze
        diffusion_analysis = morris_analyze.analyze(
            sensitivity_problem,
            parameter_values,
            diffusion_proportions,
            conf_level=0.95,
            print_to_console=True,
            num_levels=num_levels,
        )

        number_analysis = morris_analyze.analyze(
            sensitivity_problem,
            parameter_values,
            language_number,
            conf_level=0.95,
            print_to_console=True,
            num_levels=num_levels,
        )

        handle_results(number_analysis, sensitivity_problem, directory_path, "language_number")

        handle_results(diffusion_analysis, sensitivity_problem, directory_path, "diffusion")
