import matplotlib.pyplot as plt
import numpy as np
from mcmc import mcmc
from ode import AdaGradODE
from risk_ode import RiskODE
from sgd import AdaGrad

if __name__ == "__main__":
    e_var = 0
    G = 1
    b_0 = 1

    sgd = AdaGrad(G=G, b_0=b_0, e_var=e_var)
    ode = AdaGradODE("least squares", G=G, b_0=b_0, e_var=e_var)
    risk_ode = RiskODE(G=G, b_0=b_0)

    d = 4000
    T = 30

    X_ = np.random.normal(size=d)
    X_ = X_ / np.sqrt((X_ @ X_))

    eigs = np.ones(d)

    losses, steps = sgd.get_losses(eigs, T=T)
    ode_losses, ode_steps = ode.get_losses(eigs, T=T, X_=X_)
    risk_losses, risk_steps = risk_ode.get_losses(T=T)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    def t(losses): return np.linspace(0, T, len(losses))
    plt.loglog(t(losses), losses, label="AdaGrad")
    plt.loglog(t(ode_losses),
               ode_losses, label="AdaGrad ODE")
    plt.loglog(t(risk_losses), risk_losses, label="Risk ODE")
    plt.legend()
    plt.title("Losses")

    plt.subplot(1, 2, 2)
    plt.loglog(t(steps), steps, label="AdaGrad")
    plt.loglog(t(ode_steps), ode_steps, label="AdaGrad ODE")
    plt.loglog(t(risk_steps), risk_steps, label="Risk ODE")
    plt.title("Stepsizes")

    plt.suptitle(r"Comparison of AdaGrad and ODEs, identity eigs")
    plt.legend()
    plt.savefig("plots/sgd_vs_ode.png", dpi=1000)

    plt.show()
