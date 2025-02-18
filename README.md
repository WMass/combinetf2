![Framework Logo](data/logo/logo.svg)

Perform complex profile binned maximum likelihood fits by exploiting state-of-the-art differential programming. 
Computations are based on the tensorflow 2 library and scipy minimizers with multithreading support on CPU (FIXME: and GPU).
Implemted approximations in the limit of large sample size to simplify intensive computations.

## Setup

CombineTF2 can be run within a comprehensive singularity (recommended) or in an environment set up by yourself. 
It makes use of the [wums](https://pypi.org/project/wums) package for storing hdf5 files in compressed format.

### In a python virtual environment
The simplest is to make a python virtual environment. It depends on the python version you are working with (tested with 3.9.18).
First, make a python version, e.g. in the combinetf2 base directory (On some machines you have to use `python3`):
```bash
python -m venv env
```
The activate it and install the necessary packages
```bash
source env/bin/activate
pip install wums tensorflow numpy h5py hist scipy matplotlib mplhep seaborn pandas plotly kaleido
```
The packages `matplotlib`, `mplhep`, `seaborn`, `pandas`, `plotly`, and `kaleido` are only needed for the plotting scripts. 
In case you want to contribute to the development, please also install the linters `isort`, `flake8`, `autoflake`, `black`, and `pylint` used in the pre-commit hooks and the github CI
Deactivate the environment with `deactivate`.

### In singularity
(Currently not working due to missing `wums` package)

The singularity includes a comprehensive set of packages. 
It also comes with custom optimized builds that for example enable numpy and scipy to be run with more than 64 threads (the limit in the standard build).
Activate the singularity image (to be done every time before running code). 
```bash
singularity run /cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/bendavid/cmswmassdocker/wmassdevrolling\:latest
```

### Get the code
```bash
MY_GIT_USER=$(git config user.github)
git clone git@github.com:$MY_GIT_USER/combinetf2.git
cd combinetf2/
git remote add upstream git@github.com:WMass/combinetf2.git
```

Get updates from the central repository (and main branch)
```bash
git pull upstream main
git push origin main
```

### Run the code
Setting up environment variables and python path (to be done every time before running code).
```bash
source setup.sh
```

## Making the input tensor

An example can be found in ```tests/make_tensor.py -o test_tensor.hdf5```. 

### Symmetrization
By default, systematic variations are asymmetric. 
However, defining only symmetric variations can be beneficial as a fully symmetric tensor has reduced memory consumption, simplifications in the likelihood function in the fit, and is usually numerically more stable. 
Different symmetrization options are supported:
 * "average": TBD
 * "conservative": TBD
 * "linear": TBD
 * "quadratic": TBD
If a systematic variation is added by providing a single histogram, the variation is mirrored. 

## Run the fit

For example:
```bash
python scripts/fit.py test_tensor.hdf5 -o results/fitresult.hdf5 -t 0 --doImpacts --globalImpacts --binByBinStat --saveHists --computeHistErrors --project ch1 a --project ch1 b
```

## Fit diagnostics

Nuisance parameter impacts:
```bash
python scripts/printImpacts.py results/fitresult.hdf5
```

## Contributing to the code

We use pre-commit hooks and linters in the CI. Activate git pre-commit hooks (only need to do this once when checking out)
```
git config --local include.path ../.gitconfig
```
I case combineTF2 is included as a submodule, use instead:
```
git config --local include.path "$(git rev-parse --show-superproject-working-tree)/.gitconfig"
```
