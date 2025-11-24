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
PYTHONPATH=source/package python source/script/leco_model.py run configuration.toml directory
```

The run command requires a configuration file in toml format of which details can be found below. The user is also required to provide a path to a non-existing directory where the output will be stored. The standard run generates .geoparquet files of the population data, including agents ids, positions and language profiles, for every time step. A copy is created of the configuration file within the output directory.

## Configuration file

The simulation requires a TOML configuration file to set up the parameters to run the `leco` model. This file should include the following parameters:

```toml
[initialization]
agents = 5
steps = 100
seed = 51
nr_start_languages = 5

[space]
shape = [1000, 1000]

[initialization_subset_area]
present = true
x_extent = [480,520]
y_extent = [930,970]

[barrier]
present = false
x_extent = [250, 450]
y_extent = [0, 1000]
impermeability = 0.6

[population_dynamics]
death_rate = 0.01
birth_rate = 0.01
logistic_growth = true
multiplier = 50
end_growth_time = 100

[movement]
speed = 20

[language]
meanings = 100
forms = 120
mutation_rate = 0.001

[interaction]
radius = 20
partner_prob = 0.8
diffusion_rate = 0.01
```

Different scenario's can be chosen at initialization of the model. When init_subset_area is set to false, initial positions of the agents are randomly distributed across the entire space. Otherwise initial positions are confined to the subset area, for which the x and y ranges can be specified with init_x and init_y. A user can additionally specify the number of initial languages and agents. In case this value is not the same, the start languages are evenly distributed across the initial agents.

A spatial barrier can be specified by setting the ranges for both x and y values. The bar_impermeability parameter determines the degree of hinder as opposed by the barrier, ranging from 0 to 1 whereby a value of 0 means no hinder and a value of 1 complete blockage. The impermeability is used to proportionally decrease the probability of an agent to move and interact across the barrier. When an agent at first try is not allowed to pass the barrier, it will remain at it's previous position.

Population dynamics can follow constant birth and death rates as determined in the configuration file by setting the logistic_growth boolean to false. If logistic_growth is set to true, the number of agents will increase following a logistic growth curve with a constant death rate as configured. The carrying capacity K is determined by the number of agents * the multiplier as specified in the configuration file. Furthermore, the user can specify the duration of the logistic growth from the first time step until the end_growth_time step.

## Post-processing options

The `leco` package provides several options of post-processing the data. The language profiles of the agents can be classified into languages with the classify option:

```bash
cd leco
PYTHONPATH=source/package python source/script/leco_model.py cluster [--method <all|feed_forward>] [--distance <distance_threshold>] directory
```

The clustering step uses the .geoparquet files from the output directory created during the run as input and creates a .gpkg file as output containing the entire agent population across all time steps. There are two clustering methods available, of which the default is set at 'all'. The 'all' clustering method takes all language profiles across all time steps and clusters these using hierarchical clustering. The 'feed_forward' method starts from the first time step and uses diversification and shift processes from a evolutionary perspective to determine clusters of languages. The latter one is therefore more theory based. The --distance argument can be used to specify the distance threshold used for language classification. This is optional and the default is set at a value of 0.3.

Other post-processing options regard creating plots of the output data. This will generate different plots, including the number of languages over time, the number of agents per language over time, an animated plot of the agents positions over time, a 3D interactive plot and a phylogeny.

```bash
cd leco
PYTHONPATH=source/package python source/script/leco_model.py plot [--debug] configuration.toml population.gpkg
```

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
