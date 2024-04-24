import matplotlib.pyplot as plt
import numpy as np
from mcmc import mcmc
from tqdm import tqdm
from utils import rand


def get_losses(G, eigs, e, T=3, b_2=0.5**2, x_=None):
    d = len(eigs)
    cov = np.array(eigs)

    def noise():
        return e * np.random.normal()

    if x_ is None:
        x_ = rand(d)
        x_ = x_ / np.sqrt((x_ @ x_))

    x = np.zeros(d)

    def grad_f(x):
        a = rand(d, cov)
        b = a @ x_ + noise()
        return a * (a @ x - b)

    def step(x, b_2):
        g = 1/d * grad_f(x)
        b_2 = b_2 + np.linalg.norm(g)**2
        x = x - G/(np.sqrt(b_2)) * g
        return x, b_2, G/(np.sqrt(b_2))

    def loss(x):
        return 0.5 * (((x - x_) * cov) @ (x - x_) + e**2)

    losses = []
    stepsizes = []
    interval = round(d/256)

    for i in tqdm(range(T * d)):
        x, b_2, stepsize = step(x, b_2)
        if i % interval == 0:
            losses.append(loss(x))
            stepsizes.append(stepsize)

    return losses, stepsizes


if __name__ == "__main__":
    e = 0
    G = 1
    b_0 = 0.5
    T = 100
    d = 2000

    plt.figure(figsize=(12, 6))

    def dist(x): return x**(-0.25)
    eigs = mcmc(d, dist, 0, 1)

    losses, steps = get_losses(G, eigs, e, T=T, b_2=b_0**2)
    x = np.linspace(0, T-1, len(losses))

    plt.subplot(1, 2, 1)
    plt.loglog(
        x, losses, label=f"Loss, e={e}")
    plt.title("Losses")

    plt.subplot(1, 2, 2)
    plt.loglog(
        x, steps, label=f"Stepsize, e={e}")
    plt.title("Stepsizes")

    plt.suptitle(f"No noise, eigs ~ x^-1/4")
    plt.savefig("plots/sgd.png", dpi=1000)
    plt.show()
