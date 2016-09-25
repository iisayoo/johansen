# johansen
Python implementation of the Johansen test for cointegration

Installation notes:

This package requires scipy, which in turn  requires blas, lapack, atlas, and
gfortran. These can be installed on a Ubuntu system with:

    sudo apt-get install libblas-dev liblapack-dev libatlas-base-dev gfortran

Examples:

See examples folder for a jupyter notebook with example usage.

NOTE:

The cases when the chosen model (in the language of MacKinnon 1996) is 1\* or 2\* have not yet been fully implemented. They will be completed in the near future.
