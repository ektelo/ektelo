# Ektelo

Ektelo is an operator-based framework for implementing privacy algorithms.  It was first presented at SIGMOD 2018:

- Dan Zhang, Ryan McKenna, Ios Kotsogiannis, Michael Hay, Ashwin Machanavajjhala, and Gerome Miklau. 2018. [EKTELO: A Framework for Defining Differentially-Private Computations](https://dl.acm.org/citation.cfm?id=3196921). In Proceedings of the 2018 International Conference on Management of Data ([SIGMOD '18](https://sigmod2018.org)). ACM, New York, NY, USA, 115-130. DOI: https://doi.org/10.1145/3183713.3196921

In the documentation below, this is referred to as the "Ektelo paper."

Licensed under [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0.txt).

## Overview

### Architecture

There are two complementary objectives of the Ektelo project:

1. Isolate private interactions with data in a compact and secure *kernel*.
2. Modularize privacy-related algorithms into *operators*, which
promote code reuse and assist in keeping the kernel compact and secure.

The layout of the Ektelo repository reflects these goals. Code that is intended
to run on a private server is found in the module `ektelo/private`, while
non-private, client code is located in the module `ektelo/client`. We assume
that the kernel will be setup on a private server by an entity with access to
the unaltered, private data. Along with the kernel, a *kernel service*
responsible for servicing client requests must also be setup on the server. On
the client side, a privacy engineer creates a *protected data source*, which
mediates all interactions with the kernel via communication with the kernel
service.

Ektelo is designed to support interactive data queries from the privacy
engineer to the kernel. To do so, a separate kernel instance is instantiated
with a specific privacy budget for every user. At the kernel, the total
privacy expenditure is tracked for each query according to Algorithm 6 in the
Ektelo paper. User queries are serviced until the budget has been exceeded.
At that point, a `BudgetExceeded` error is sent back to the user.

### Examples

1. File `examples/cdf_estimator.py` provides an example of the entire Ektelo workflow.  This example aligns with Algorithm 1 from the Ektelo paper.
2. File `examples/standalone_plan.py` provides an example of a previously published algorithm expressed as an Ektelo *plan* consisting of a sequence of Ektelo *operators*.  The algorithm in this case is MWEM (Hardt et al. ["A Simple and Practical Algorithm for Differentially Private Data Release."](http://papers.nips.cc/paper/4548-a-simple-and-practical-algorithm-for-differentially-private-data-release) NIPS 2012).  Note this example *excludes* the layer that manages the interaction between client code and the protected kernel.  While removing this layer makes it easier to trace the plan, it also removes the privacy protection (i.e., the variable `R` corresponds to the input dataset so adding `print(R)` would result in full disclosure of the "private" input).  We imagine that writing Ektelo plans in this "stripped down" form may be useful for privacy researchers who are designing new algorithms and only executing on non-sensitive inputs.
3. File `examples/private_plan.py` is the same as the previous example (`standalone_plan.py`) except that it *includes*  the layer that manages client-kernel interaction.  In this example, any interactions with the private data are mediated by the kernel, which will ensure protection.  In particular, the `R` variable is now a `ProtectedDataSource` and invoking a method on `R` will trigger an interaction with the kernel.  This example illustrates how a complex differentially private algorithm can be executed via client calls to the protected kernel.
4. File `examples/budget_exceeded.py` provides an example of a client-kernel
interaction that produces such a `BudgetExceeded` error.


Examples 2 and 3 above illustrate the MWEM algorithm written as an Ektelo plan.   Other algorithms from the literature have also been written as plans in two places: `plans/standalone.py` and `plans/private.py`.  The standalone plans exclude the client-kernel layer (similar to example 2 above) and the private plans include it (similar to example 3 above).

## Setup

### Example Environment

```bash
export EKTELO_HOME=$HOME/Documents/ektelo
export EKTELO_DATA=/tmp/ektelo
export PYTHON_HOME=$HOME/virtualenvs/PyEktelo
export PYTHONPATH=$PYTHONPATH:$EKTELO_HOME
export EKTELO_LOG_PATH=$HOME/logs
export EKTELO_LOG_LEVEL=DEBUG
```

### System dependencies

Various system-level packages are necessary to meet the requirements
for third-party python modules installed during initialization. The
dependencies vary by platform.

#### Ubuntu 16.04 Packages

```bash
sudo apt-get install python3.4-venv gfortran liblapack-dev libblas-dev
sudo apt-get install libpq-dev python3-dev libncurses5-dev swig glpk
```

#### OSX packages

```bash
brew install swig
```

### Initialization

Be sure to setup the environment (describe above) first. You will need to
install several packages. The following commands should work for debian systems.

Next, create a virtual environment for python by entering the commands below.

```bash
mkdir $EKTELO_LOG_PATH
python3 -m venv $PYTHON_HOME
source $PYTHON_HOME/bin/activate
cd $EKTELO_HOME
pip install -r resources/requirements.txt
```

The data must be downloaded into the `$EKTELO_DATA` folder.

```bash
mkdir -p $EKTELO_DATA
curl https://www.dpcomp.org/data/cps.csv > $EKTELO_DATA/cps.csv
curl https://www.dpcomp.org/data/stroke.csv > $EKTELO_DATA/stroke.csv
```

Finally, after instantiating the virtualenv, compile the C libraries as follows.

```bash
cd $EKTELO_HOME/ektelo/algorithm
./setup.sh
```

### Session

Once initialization has been run, the virtual environment can be restored with
the following command.

```bash
source $PYTHON_HOME/bin/activate
```

### Testing

Execute the following in the base of the repository.

```bash
cd $EKTELO_HOME
nosetests
```
To test a specific module (in this case, `TestExperiment`):

```bash
nosetests test.unit.test_data:TestData
```
