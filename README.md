# Leco

**L**inguistic **E**volutionary **CO**mputations

## Develop

After cloning the repository, a virtual environment can be initialized and activated which contains the
required 3rd-party packages. For this there are multiple options, like Python's built-in `venv` package or
Conda.

### Using `venv`

```bash
cd leco

# Create and initialize a virtual environment
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r environment/configuration/requirements.txt
pre-commit install
```

Once the development environment has be setup, all that is needed to start developing is:

```bash
cd leco

source .venv/bin/activate
```



### Using Conda

TODO

## Test

Code for testing the `leco` Python package can be put in `source/test/test_leco` directory. The command for
running the tests is:

```bash
cd leco
PYTHONPATH=source/package python source/test/run_all_tests.py
```



## Run

To run the `leco` model, type this command:

```bash
cd leco
PYTHONPATH=source/package python source/script/leco_model.py run [--start_geoparquet steps201-400.geoparquet] [--intermediate_step 350] configuration.toml directory
```

The run command requires a configuration file in toml format of which details can be found below. The user is also required to provide a path to a non-existing directory where the output will be stored. The standard run generates .geoparquet files of the population data, including agents ids, positions and language profiles, for every write_interval number of time steps, as defined in the configuration file. A copy is created of the configuration file within the output directory.

Instead of initializing a specific number of languages at the start of the run, the user can choose to take another run at any point in time as a starting point. This can be indicated by defining which geoparquet file, with `--start_geoparquet`, and which time step, with `--intermediate_step`, to start from.

### Configuration file

The simulation requires a TOML configuration file to set up the parameters to run the `leco` model. This file should include the following parameters:

```toml
[initialization]
agents = 5
steps = 20
seed = 51
write_interval = 200
nr_start_languages = 5

[space]
shape = [1000, 1000]

[initialization_subset_area]
present = false
x_extent = [480,520]
y_extent = [930,970]

[barrier]
present = false
x_extent = [250, 450]
y_extent = [0, 1000]
impermeability = 0.6

[movement]
speed = 20

[language]
meanings = 100
forms = 120
mutation_rate = 0.001

[population_dynamics]
death_rate = 0.01
birth_rate = 0.01
logistic_growth = true
carrying_capacity = 10
end_growth_time = 50

[interaction]
radius = 20
partner_proportion = 0.8
diffusion_rate = 0.01
similarity_preference = 0.2
```

The same file is used as input for the cluster command.

Population dynamics can follow constant birth and death rates as determined in the configuration file by setting the logistic_growth boolean to false. If logistic_growth is set to true, the number of agents will increase to carrying capacity within a time period of `end_growth_time` following a logistic growth curve with a constant death rate as configured. For further clarification on the parameters, please refer to *the paper*.

## Post-processing

As a post-processing step, the language profiles of the agents can be classified into languages with the cluster command:

```bash
cd leco
PYTHONPATH=source/package python source/script/leco_model.py cluster [--linkage average] [--distance 0.3] [--start_gpkg population_base.gpkg] [--intermediate_step 350] configuration.toml directory population.gpkg
```

The cluster command uses the .geoparquet files that are located in the defined directory and which were created during the run, as input and creates a .gpkg file as output containing the entire agent population across all time steps. This command makes use of the *AgglomerativeClustering* function of the scikit-learn package [AgglomerativeClustering — scikit-learn 1.9.0 documentation](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html), which is a form of unsupervised clustering. The user can define the linkage (default: average) and the distance threshold (default: 0.3) used for the clustering. When a run uses `--start_geoparquet` and `--intermediate_step`, pass the same time step to cluster via `--intermediate_step`, together with the warm-up run’s clustered output via `--start_gpkg`. Clustering then continues from those existing language assignments instead of starting over at time step 0.

In addition, the `leco` package includes scripts to visualize the outcomes of a single run with the plot command:

```bash
cd leco
PYTHONPATH=source/package python source/script/leco_model.py plot configuration.toml directory population.gpkg
```

This will generate different plots, including the number of languages over time, the number of agents per language over time, a phylogeny of the different language families, and the spatial distribution of languages and families at different points in time.

## Spawn

To run multiple runs in parallel, the spawn command can be used for the run and cluster commands as follows.

```bash
cd leco
PYTHONPATH=source/package python source/script/leco_spawn.py run --max_nr_workers=8 spawn.toml [-- additional arguments]
```

```bash
cd leco
PYTHONPATH=source/package python source/script/leco_spawn.py cluster --max_nr_workers=8 spawn.toml [-- additional arguments]
```



### Spawn configuration

The spawn.toml configuration file is organized as follows.

```toml
# Regular Leco model configuration file
configuration_file = "configuration.toml"

# Template to use for generating output directory pathnames. All variable parameters must be mentioned.
directory_pattern = "~/tmp/spawn/speed_{speed}/mutation_rate_{mutation_rate}/radius_{radius}/diffusion_rate_{diffusion_rate}/similarity_preference_{similarity_preference}/seed_{seed}"

[sensitivity.run]
# Section with settings for performing a sensitivity analysis
method = "ofat" # One factor at a time

[sensitivity.run.initialization]
seed.range = [51, 53, 1]

[sensitivity.run.movement]
speed.range = [12, 22, 2]

[sensitivity.run.language]
mutation_rate.range = [0.0001, 0.0005, 0.0001]

[sensitivity.run.interaction]
radius.range = [10, 50, 10]
diffusion_rate.range = [0.05, 0.1, 0.01]
similarity_preference.range = [-1.0, 1.2, 0.2]
```

The spawn command is designed for running sensitivity analyses, during which the values of different parameters are systematically varied. Currently, the one-factor-at-a-time sensitivity analysis is the single method that is incorporated, meaning that only one of the parameters is varied at a time. Baseline values are taken from the regular `leco` configuration file that is defined in the top line. The range within which parameters are varied, are defined in the spawn.toml, as [minimum, maximum, step].

## Analysis of multiple runs

The results of multiple `leco` runs and classifications can be summarized using the script `leco_summarized.py`.

```bash
cd leco
PYTHONPATH=source/package python environment/script/leco_summarized.py directory [--baseline]
```

This summarizes multiple output metrics across different parameter combinations as created by `spawn`. When setting the `--baseline` argument, it only summarizes the output metrics for different seeds of a single parameter combination. The metrics are written to a csv file.

The summarized values can be visualized with the `summarized_plots.py`.

```bash
cd leco
PYTHONPATH=source/package python environment/script/summarized_plots.py summarized.csv [--baseline] [--classification single:single_summarized.csv complete:complete_summarized.csv]
```

This will generate a spider plot of the effect of different parameters on the number of languages, as well as several plots to visualize the number of languages and speaker distribution for the baseline parameter combination. When setting the `--baseline` argument, and providing the summarized .csv file generated with the  `--baseline` argument in `leco_summarized.py,` it will generate summary tables for the output metrics, plots of these metrics over time, and baseline plots on language number and speaker distribution. Alternatively, the `--classification` argument allows for comparison of multiple summarized files created with the `--baseline` argument. This way, the outputs of a single parameter combination for different language classification configurations can be compared. The command above shows an example of how to compare the summarized files of two different types of linkages, by giving first the name and then the path to the csv file. The `--classification` argument generates a summary table and a plot showing the number of languages over time.

## Create wheel file

To create a `leco` wheel file, type this command:

```bash
cd leco
PYTHONPATH=source/package python -m build --wheel --outdir $HOME/tmp/dist .
```

The resulting wheel file can be installed like this:

```bash
pip3 install -f $HOME/tmp/dist leco
```



## Generate documentation

Leco API documentation can be generated like this:

```bash
cd leco
sphinx-build -M html documentation $HOME/tmp/documentation
```

Point your browser at `$HOME/tmp/documentation/index.html`.
