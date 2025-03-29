![Framework Logo](data/logo/logo.svg)

Perform complex profile binned maximum likelihood fits by exploiting state-of-the-art differential programming. 
Computations are based on the tensorflow 2 library and scipy minimizers with multithreading support on CPU (FIXME: and GPU).
Implemented approximations in the limit of large sample size to simplify intensive computations.

## Install

You can install combinetf2 via pip. It can be installed with the core functionality:
```bash
pip install combinetf2
```
Or with optional dependencies to use the plotting scripts
```bash
pip install combinetf2[plotting]
```


### Get the code

If you want to have more control or want to develop CombineTF2 you can check it our as (sub) module.

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

It can be run within a comprehensive singularity (recommended) or in an environment set up by yourself. 
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
pip install wums[pickling,plotting] tensorflow tensorflow-probability numpy h5py hist scipy matplotlib mplhep seaborn pandas plotly kaleido
```
The packages `matplotlib`, `mplhep`, `seaborn`, `pandas`, `plotly`, and `kaleido` are only needed for the plotting scripts. 
In case you want to contribute to the development, please also install the linters `isort`, `flake8`, `autoflake`, `black`, and `pylint` used in the pre-commit hooks and the github CI
Deactivate the environment with `deactivate`.

### In singularity
The singularity includes a comprehensive set of packages. 
But the singularity is missing the `wums` package, you have to check it our as a submodule.
It also comes with custom optimized builds that for example enable numpy and scipy to be run with more than 64 threads (the limit in the standard build).
Activate the singularity image (to be done every time before running code). 
```bash
singularity run /cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/bendavid/cmswmassdocker/wmassdevrolling\:latest
```

### Run the code
Setting up environment variables and python path (to be done every time before running code).
```bash
source setup.sh
```

## Making the input tensor

An example can be found in ```tests/make_tensor.py -o test_tensor.hdf5```. 

### Systematic uncertainties
Systematic uncertainties are implemented by default using a log-normal probability density (with a multiplicative effect on the event yield).  Gaussian uncertainties with an additive effect on the event yield can also be used.  This is configured through the `systematic_type` parameter of the `TensorWriter`.

### Sparse tensor
By setting `sparse=True` in the `TensorWriter` constructor the tensor is stored in the sparse representation. 
This is useful when working with a sparse tensor, e.g. having many bins/processes/systematics where each bin/process/systematic only contributes to a small number of bins/processes/systematics. 
This is often the case in the standard profile likelihood unfolding. 

### Symmetrization
By default, systematic variations are asymmetric. 
However, defining only symmetric variations can be beneficial as a fully symmetric tensor has reduced memory consumption, simplifications in the likelihood function in the fit, and is usually numerically more stable. 
Different symmetrization options are supported:
 * "average": TBD
 * "conservative": TBD
 * "linear": TBD
 * "quadratic": TBD
If a systematic variation is added by providing a single histogram, the variation is mirrored. 

### Masked channels
Masked channels can be added that don't contribute to the likelihood but are evaluated as any other channel. 
This is done by defining `masked=True` in the `tensorwriter` `add_channel` function. 
(Pseudo) Data histograms for masked channels are not supported.
This is useful for example to compute unfolded (differential) cross sections and their uncertainties, including global impacts, taking into account all nuisance parameters that affect these channels.

## Run the fit

For example:
```bash
combinetf2_fit test_tensor.hdf5 -o results/fitresult.hdf5 -t 0 --doImpacts --globalImpacts --binByBinStat --saveHists --computeHistErrors
```

### Bin-by-bin statistical uncertainties
Bin-by-bin statistical uncertainties on the templates can be added at runtime using the `--binByBinStat` option.  The Barlow-Beeston lite method is used to add implicit nuisance parameters for each template bin.  By default this is implemented using a gamma distribution for the probability density, but Gaussian uncertainties can also be used with `--binByBinStatType normal`.

### Physics models
Physics models are used to perform transformation on the observables (the histogram bins in the (masked) channels). 
They are defined `combinetf2/physicsmodels.py` and can be called in `combinetf2_fit` with the `--PhysicsModel` or `-m` option e.g. ` -m project ch1 a -m project ch1 b`. 
The first argument is the physics model name followed by arguments passed into the physics model.
Custom physics models can be added to make the desired transformation. Available physics models are
 * "project": To project histograms to lower dimensions, respecting the covariance matrix across bins.
 * "normalize": To normalize histograms to their sum (and project them) e.g. to compute normalized differential cross sections.
 * "ratio": To compute the ratio between channels, processes, or histogram bins.
 * "normratio": To compute the ratio of normalized histograms.

## Fit diagnostics

Nuisance parameter impacts:
```bash
combinetf2_print_impacts results/fitresult.hdf5
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
