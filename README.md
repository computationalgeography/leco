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
PYTHONPATH=source/package python source/script/leco_model.py -i path/config.toml -o outputpath
```

-i requires a configuration file in TOML format of which details can be found below.
-o specifies the path to where the ouput will be stored.

The standard run generates .geoparquet files of the population data, including agents ids, positions and language profiles, for every timestep. In addition, a parameter.txt file is created in the same output directory containing the configuration parameters from the config.toml file.

## Configuration file

The simulation requires a TOML configuration file to set up the parameters to run the `leco` model. This file should include the following parameters:

```toml
# Initialization settings
agents = 5
steps = 200
seed = 42
nr_start_languages = 5
init_subset_area = true
init_x = [480,520]
init_y = [930,970]

# Spatial boundaries
x_max = 1000
y_max = 1000

# Barrier
barrier = false
bar_x = [250, 450]
bar_y = [0, 1000]
bar_impermeability = 0.6

# Agent attributes
speed = 20
meanings = 100
forms = 120
mutation_rate = 0.001

# Population dynamics
death_rate = 0.01
birth_rate = 0.01
logistic_growth = true
multiplier = 50
end_growth_time = 100

# Interaction settings
int_radius = 20
int_partner_prob = 0.8
diffusion_rate = 0.01
```

Different scenario's can be chosen at initialization of the model. When init_subset_area is set to false, initial positions of the agents are randomly distributed across the entire space. Otherwise initial positions are confined to the subset area, for which the x and y ranges can be specified with init_x and init_y. A user can additionally specify the number of initial languages and agents. In case this value is not the same, the start languages are evenly distributed across the initial agents.

A spatial barrier can be specified by setting the ranges for both x and y values. The bar_impermeability parameter determines the degree of hinder as opposed by the barrier, ranging from 0 to 1 whereby a value of 0 means no hinder and a value of 1 complete blockage. The impermeability is used to proportionally decrease the probability of an agent to move and interact across the barrier. When an agent at first try is not allowed to pass the barrier, it will remain at it's previous position.

Population dynamics can follow constant birth and death rates as determined in the configuration file by setting the logistic_growth boolean to false. If logistic_growth is set to true, the number of agents will increase following a logistic growth curve with a constant death rate as configured. The carrying capacity K is determined by the number of agents * the multiplier as specified in the configuration file. Furthermore, the user can specify the duration of the logistic growth from the first time step untill the end_growth_time step.

## Post-processing options

The `leco` package provides several options of post-processing the data. The language profiles of the agents can be classified into languages with the classify option:

```bash
cd leco
PYTHONPATH=source/package python source/script/leco_model.py --classify outputpath/resultsdir -d 0.2
```

The -d argument can be used to specify the distance threshold used for language classification. This is optional and the default is set at a value of 0.2. The output is stored in a single .gpkg file.

Alternatively, classification of the language profiles can be run directly after running the `leco` model:

```bash
cd leco
PYTHONPATH=source/package python source/script/leco_model.py --classify-after -i path/config.toml -o outputpath -d 0.2
```

Other post-processing options regard creating plots of the output data. The first script creates summary plots of the output data, including the number of languages over time and the number of agents per language over time, both saved as .pdf files.

```bash
cd leco
PYTHONPATH=source/package python source/script/leco_model.py --plot-summaries outputpath/resultsdir/population.gpkg
```

The second script creates an animated plot of the agents positions colored by language over time. The output is saved as a .gif file.

```bash
cd leco
PYTHONPATH=source/package python source/script/leco_model.py --plot-animation outputpath/resultsdir/population.gpkg
```

To run the `leco` model, classification and both plot steps in one go:

```bash
cd leco
PYTHONPATH=source/package python source/script/leco_model.py --all -i path/config.toml -o outputpath
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
