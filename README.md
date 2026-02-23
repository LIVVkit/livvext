![](https://raw.githubusercontent.com/wiki/LIVVkit/LIVVkit/imgs/livvkit.png)

# LIVVkit Extensions (LIVVext)

This repository holds a collection of extensions to
[LIVVkit](https://livvkit.github.io/Docs/index.html) for validation and
custom analyses of ice sheet models and their associated Earth system
models.

LIVVext (LEX) was first described in Evans, et al., 2019[^1]

## Dependencies

The Python dependencies are listed in requirements.txt and requirements-dev.txt but
this suite depends on a set of reanalysis and observational datasets,
which are part of E3SM-Diags, available on several DOE supported
machines, including Perlmutter at NERSC, and Chrysalis at ANL\'s LCRC.

The Python package itself is described in `pyproject.toml`, which is used by
`pip` to install this package

Currently, LIVVext is designed to run on NERSC's Perlmutter, and ANL-LCRC's Chrysalis,
but future work is planned to support other machines where E3SM runs.

## Environment setup

For setting up an environment to which LIVVext and dependencies will be
installed, Pixi and conda are documented here. **NB** this
will only currently work on Perlmutter and Chrysalis, the environment should be
created there.

### Pixi environment
[pixi](https://pixi.sh/latest/) is a package management tool, and the primary
environment management tool for LIVVext.


First, pixi must be installed locally following [these instructions](https://pixi.prefix.dev/latest/installation/),
then an enviornment for LIVVext development can be created:
```bash
$ git clone https://github.com/LIVVkit/LIVVext.git
$ cd LIVVext
$ pixi install          # This will install the default environment
$ pixi shell -e default # To activate the default (runtime) environment
```

If development tools are needed (e.g. `pytest`), install and activate the `dev` environment:
```bash
$ pixi install -e dev   # Installs the environment
$ pixi shell -e dev     # Activates the dev environment (for code testing and checks)
```
**N.B.** it is not recommended to do `pixi install --all`, as this will install the environment versions used for testing.

### Conda environment
```bash
$ git clone https://github.com/LIVVkit/LIVVext.git
$ cd LIVVext
$ {conda, mamba} env create -n lex_env python --file env.yml
$ {conda, mamba} activate lex_env
$ pip install -e .    # Installs the LIVVext module as an editable Python package to the lex_env environment.
```

## Basic usage

Within the `LIVVext/config` directory there are templates for
ELM r05 and r025 resolutions, as well as pre-existing configurations for
several current runs.

To execute any of these analyses, point `livv` (LIVVkit\'s command line
interface) to any of these extensions via the the `-V/--validate`
option.

For example, to run the minimal example extension, place the output
website in the `vv_test` directory, and serve the output
website you\'d run this command:

```bash
$ livv -V config/example/example.yml -o vv_test -s
```
This will create a directory in the current directory called `vv_test` (~7.5 MB), and
spawn an HTTP server, which should only be used for testing purposes.
(This works best if the output is in the current directory)

*Note:* All the extension configurations files assume you are working
from the top level `LIVVext` directory. You *can* run any of these
extensions from any directory, but you will need to edit the paths in
the YAML configuration files so that `livv` can find the required files.

Likewise, you can also apply these analyses to any new model run[^2] by
adjusting the paths to point to your model run.

## Running existing cases on PM-CPU

The `LIVVext/run_livv.sh` script will run all the currently
available analyses on pm-cpu for a particular case, e.g.:
```bash
$ cd $HOME/LIVVext
$ ./run_livv.sh v2.1.r025.IGERA5ELM_MLI-deep_firn_1980_2020
```
Will create a web output at `/global/cfs/projectdirs/e3sm/www/${USER}/v2.1.r025.IGERA5ELM_MLI-deep_firn_1980_2020`,
viewable at https://portal.nersc.gov/project/e3sm/${USER}/v2.1.r025.IGERA5ELM_MLI-deep_firn_1980_2020

The batch script provided will run all current cases on Perlmutter on a
compute node in parallel
```bash
$ cd LIVVext
$ sbatch run_lex_pm-cpu.sbatch
```

## Running new cases on PM-CPU

### Generate a single timeseries file from ELM h0 outputs
- `CASE="The case name"`
- `ncrcat -v topo,landfrac,QSNOFRZ,FSRND,FSRVD,FSDSVD,FSDSND,EFLX_LH_TOT,FIRA,FLDS,FSA,FSDS,FSH,QICE,QRUNOFF,QSNOMELT,QSOIL,RAIN,SNOW,TSA,SNOWICE,SNOWLIQ,H2OSNO ${CASE}.elm.h0*.nc -o ${CASE}.nc`

### Perform post-processing on a single time series ELM h0 output
- Edit the `LIVVext/LIVVext/postproc/e3sm/postproc.sbatch` batch file to mach the new run

    Key variables:
    - `INDIR`: Path which contains single output time series file
    - `OUTCASE`: Name of the new case which is the name of the netCDF file without
        extension (e.g. `v2.1.r025.IGERA5ELM_MLI-deep_firn_1980_2020`)
    - `RES`: ELM output resolution (currently accepts `R05` and `R025`)
    - `OUTDIR`: Scratch directory into which climatology files will be written,
        defaults to `${SCRATCH}/LIVVext/data/e3sm/${OUTCASE}`
- Run the post-processing script:
  - `cd LIVVext/LIVVext/postproc/e3sm; sbatch postproc.sbatch`

**NB**: the `postproc.sbatch` script will create the configuration for your case
(based on `OUTCASE` and `OUTDIR`), then run LIVVkit on it with `LIVVext/run_livv.sh`

## Developing a custom extension

See the [LIVVkit documentation](https://livvkit.github.io/Docs/LIVVext.html)
for details on how to develop an extension. Briefly, a absolute minimum
working example is provided by the `example/` extension, which should
be copied to provide the basis for your new extension. All extensions
are required to contain a minimal working example set of data such that
they can be run an executed on any machine.

For extensions that require data for which re-host permission cannot be
granted, they must include documentation on how to acquire and use the
data as well as either a small set of processed data or a set of
\"fake\" example data.

## Issues, questions, comments, etc.?

If you would like to suggest features, request tests, discuss
contributions, report bugs, ask questions, or contact us for any reason,
use the [LIVVkit issue
tracker](https://github.com/LIVVkit/LIVVkit/issues). [LIVVext issue
tracker](https://github.com/LIVVkit/livvext/issues).

Want to send us a private message?

**Michael E. Kelleher** :github: \@mkstratos

**Joseph H. Kennedy** :github: \@jhkennedy

**Katherine J. Evans** :github: \@kevans32

[^1]:   Evans, K.J., J.H. Kennedy, D. Lu, M.M. Forrester, S. Price, J. Fyke,
    A.R. Bennett, M.J. Hoffman, I. Tezaur, C.S. Zender, and M. Vizcaino
    (2019). LIVVkit 2.1: Automated and extensible ice sheet model
    validation. *Geoscientific Model Development.* https://gmd.copernicus.org/articles/12/1067/2019/

[^2]: This assumes the new data files conform to the format of the
    included data files. That is, an extension that analyses output from
    the CISM-Albany ice sheet model will likely be able to analyze any
    similar CISM-Albany simulation, but likely would *not* be able to
    analyze output from the PISM ice sheet model without \"massaging\"
    the PISM files into a CISM-Albany like structure, or adjusting the
    extension. *This is a problem we are working on for future LIVVext 
    releases.*
