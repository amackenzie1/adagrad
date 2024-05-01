import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from utils import rand
from with_numba import gauss_hermite


def R(x, x_):
    return -x * np.exp(x_) / (1 + np.exp(x_)) + np.log(np.exp(x) + 1)


def get_losses(G, eigs, x_, T=3):

    d = len(eigs)
    cov = np.array(eigs)

    x = np.zeros(d)
    y = np.zeros(d)

    def grad_f(x):
        a = rand(d, cov)
        ex = np.exp(a @ x)
        ex_ = np.exp(a @ x_)
        return a * (ex / (1 + ex) - ex_ / (1 + ex_))

    def step(x, y, b_2):
        g = 1 / d * grad_f(x)
        b_2 = b_2 + np.linalg.norm(g) ** 2
        x = x - G / (np.sqrt(b_2)) * g
        return x, y, b_2

    def loss(x):
        B = np.array([[(x * cov) @ x, (x * cov) @ x_],
                     [(x * cov) @ x_, (x_ * cov) @ x_]])
        return gauss_hermite(B, "R_lg")

    losses = []
    exes = []
    bees = []
    interval = round(d / 256)
    b_2 = (0.5) ** 2

    for i in tqdm(range(T * d)):
        x, y, b_2 = step(x, y, b_2)
        if i % interval == 0:
            losses.append(loss(x))
            exes.append(x)
            bees.append(b_2)
    return losses


if __name__ == "__main__":

    sizes = [500, 1000, 5000, 10000]
    G = 1.5
    eigs = [1]

    for d in sizes:
        x_ = rand(d)
        x_ = x_ / np.sqrt((x_ @ x_))
        losses = get_losses(G, eigs * d, x_)

        x = np.linspace(0, int(len(losses) / 256), len(losses))
        plt.plot(losses, label=f"Dimension {d}")

    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Loss")
    plt.title("Losses for logistic regression")
    plt.savefig("plots/logistic.png", dpi=1000)
    plt.show()
