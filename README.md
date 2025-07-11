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
PYTHONPATH=source/package python source/script/leco_model.py -i path/config.toml -o path
```

-i requires a configuration file in TOML format of which details can be found below.
-o is an optional argument specifying the path to where the ouput will be stored. If -o is not given, no output will be stored.

## Configuration file

The simulation requires a TOML configuration file to set up the parameters to run the leco model. This file should include the following parameters:

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
similarity = 0.69

# Output and visualization
plot_step = 50
```

## Create wheel file

To create a leco wheel file, type this command:

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
