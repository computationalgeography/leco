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
agents = 1000
steps = 100
seed = 42
nr_start_languages = 40

# Spatial boundaries
x_max = 1000
y_max = 1000

# Agent attributes
speed = 10
meanings = 100
forms = 120
mutation_rate = 0.001

# Population dynamics
birth_rate = 0.01
death_rate = 0.01

# Interaction settings
int_radius = 5
int_partner_prob = 0.8
diffusion_rate = 0.01
```

## Post-processing options

The `leco` package provides several options of post-processing the data. The language profiles can be classified into languages with the classify option:

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

Other post-processing options regard creating plots of the output data. At the moment, there are two scripts. The first script creates summary plots of the output data, including the number of languages over time and the number of agents per language over time, both saved as .pdf files.

```bash
cd leco
PYTHONPATH=source/package python source/script/leco_model.py --plot-summaries outputpath/resultsdir/population.gpkg
```

The second script creates an animated plot of the agents positions colored by language over time. The output is saved as a .gif file.

```bash
cd leco
PYTHONPATH=source/package python source/script/leco_model.py --plot-animation outputpath/resultsdir/population.gpkg
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
