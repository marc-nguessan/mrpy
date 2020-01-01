# mrpy

A Python library to solve evolutionary Partial Differential Equations with space adaptive multiresolution. This grid adaptation technique is based upon the work of [Cohen et al](https://dl.acm.org/doi/10.1090/S0025-5718-01-01391-6), and the specific strategy that we apply follows the tutorial devised by [Tenaud & Duarte](https://hal.archives-ouvertes.fr/hal-00697705).

## Installation

mrpy can be installed on virtually any computer with Python 3.5 or above.
The required mrpy dependencies are listed below.

### Required packages

-Cython
-h5py
-mpi4py
-numpy
-petsc4py
-scipy

Once you have cloned the git repository, you need to compile the Cython code used for the multiresolution operations. For that, go to the folder [mr_utils](./mrpy/mr_utils) and run the following line of code in the command line:

    $ python setup.py build_ext --inplace

