import matplotlib.pyplot as plt
import numpy as np


class RiskODE:
    def __init__(self, G=1, b_0=1):
        self.G = G
        self.b_0 = b_0
        self.dt = 1/256

    def rk4_system(self, dydt, t_span, y0, h):
        t0, tf = t_span
        n = int((tf - t0) / h)
        t = np.linspace(t0, tf, n+1)
        y = np.zeros((n+1, len(y0)))
        y[0] = y0

        # Runge-Kutta 4th Order Method
        for i in range(n):
            k1 = h * np.array(dydt(t[i], y[i]))
            k2 = h * np.array(dydt(t[i] + 0.5 * h, y[i] + 0.5 * k1))
            k3 = h * np.array(dydt(t[i] + 0.5 * h, y[i] + 0.5 * k2))
            k4 = h * np.array(dydt(t[i] + h, y[i] + k3))
            y[i+1] = y[i] + (k1 + 2*k2 + 2*k3 + k4) / 6

        return t, y

    def system_of_eqns(self, t, y, G, b_0):
        y1, y2 = y
        dy1dt = y2
        dy2dt = (G**2 * y2) / (b_0**2 + 2 * y1) - \
            (2 * G * y2) / np.sqrt(b_0**2 + 2 * y1)
        return [dy1dt, dy2dt]

    def get_losses(self, T=10):
        t_span = (0, T)
        t, y = self.rk4_system(lambda t, y: self.system_of_eqns(
            t, y, self.G, self.b_0), t_span, [0, 0.5], self.dt)

        return y[:, 1], self.G/np.sqrt(self.b_0**2 + 2 * y[:, 0])


if __name__ == "__main__":
    # Example usage
    ode = RiskODE()
    t, loss, step = ode.get_losses()
