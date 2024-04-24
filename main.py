import matplotlib.pyplot as plt
import numpy as np
from mcmc import mcmc
from ode import AdaGradODE
from sgd import AdaGrad

if __name__ == "__main__":
    e_var = 0
    G = 1
    b_0 = 0.5
    sgd = AdaGrad(G=G, b_0=b_0, e_var=e_var)
    ode = AdaGradODE("least squares", G=G, b_0=b_0, e_var=e_var)

    d = 2000
    T = 10

    X_ = np.random.normal(size=d)
    X_ = X_ / np.sqrt((X_ @ X_))

    def dist(x): return x**(-0.25)
    eigs = mcmc(d, dist, 0, 1)

    losses, steps = sgd.get_losses(eigs, T=T)
    ode_losses, ode_steps = ode.get_losses(eigs, T=T, X_=X_)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(np.linspace(0, T, len(losses)), losses, label="AdaGrad")
    plt.plot(np.linspace(0, T, len(ode_losses)),
             ode_losses, label="AdaGrad ODE")
    plt.title("Losses")

    plt.subplot(1, 2, 2)
    plt.plot(np.linspace(0, T, len(steps)), steps, label="AdaGrad")
    plt.plot(np.linspace(0, T, len(ode_steps)), ode_steps, label="AdaGrad ODE")
    plt.title("Stepsizes")

    plt.suptitle(r"Comparison of AdaGrad and AdaGrad ODE, eigs ~ $x^{-1/4}$")
    plt.legend()
    plt.savefig("plots/sgd_vs_ode.png", dpi=1000)

    plt.show()
