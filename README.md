The `ode.py` file contains code for the odes, linear or logistic regression; the `sgd_classic.py` and `logistic_classic.py` contain the actual reference implementations of the algorithms themselves, which we compare the ODE against.

The `mcmc.py` contains a basic implementation of MCMC sampling so that we can get eigenvalues according to whatever law we want.

Most files should contain a `__main__` that demonstrates the usage. Plots are typically saved to the `plots/` directory.
