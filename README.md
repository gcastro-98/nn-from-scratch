# nn-from-scratch
Neural network implementation from scratch through a Keras-like 
API in Python.

It is currently a little sketch, thus it just supports:
- Few loss functions: log-loss
- Few activation functions: ReLu & identity
- Few layer types: just fully-connected layers.
- The gradient descent is purely stochastic (``batch_size`` = 1) and implemented using Automatic Differentiation
  - Specifically, forward-propagation is used and implemented using the ``autograd`` package. 

As per example, the code is applied to solve a classification problem,
found at the ``example.py`` module.

## Installation
To install this package's modules into your conda environment ``conda-env``,
the .toml file can be leveraged by
```console
(conda-env) $ pip install .
```

## Development (conda) environment

The following needs to be executed in any terminal:
```console
$ conda create -n nn-dev python=3.9 -y
$ conda activate nn-dev
$ conda install numpy sklearn -y
$ conda install -c conda-forge autograd -y
$ conda install -c anaconda sphinx numpydoc \
    sphinx_rtd_theme recommonmark python-graphviz -y
$ pip install --upgrade myst-parser
```

## Documentation

Whenever the modules have been updated, the documentation can be re-generated 
from the ``docs`` folder by typing ():
```console
(nn-dev) nn-from-scratch/docs $ make html
```