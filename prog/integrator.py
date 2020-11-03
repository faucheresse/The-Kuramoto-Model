import numpy as np


class Integration:
    """Integration by Euler's, RK2's, RK4's methods"""
    def euler(self, f, x0, t):
        print("----- integrate -----")

        x = np.zeros((len(x0), len(t)))
        x[:, 0] = x0[:]

        for n in range(len(t)-1):
            deltaT = t[n + 1] - t[n]
            x[:, n + 1] = x[:, n] + f(x, t + deltaT / 2)[:, n] * deltaT

        return x

    def RK2(self, f, x0, t):
        print("----- integrate -----")

        x = np.zeros((len(x0), len(t)))
        x[:, 0] = x0[:]

        for n in range(len(t)-1):
            deltaT = t[n + 1] - t[n]
            step = x[:] + f(x, t + deltaT / 2)[:] * deltaT / 2
            x[:, n + 1] = x[:, n] + f(step, t + deltaT / 2)[:, n] * deltaT

        return x

    def RK4(self, f, x0, t):
        print("----- integrate -----")

        K = np.zeros((4, len(x0), len(t)))

        x = np.zeros((len(x0), len(t)))
        x[:, 0] = x0[:]

        for n in range(len(t)-1):
            deltaT = t[n + 1] - t[n]
            K[0, :, n] = f(x, t)[:, n]
            K[1, :, n] = f(x + K[0, :] * deltaT / 2, t + deltaT / 2)[:, n]
            K[2, :, n] = f(x + K[1, :] * deltaT / 2, t + deltaT / 2)[:, n]
            K[3, :, n] = f(x + K[2, :] * deltaT, t + deltaT)[:, n]

            x[:, n + 1] = x[:, n] + (K[0, :, n] + 2 * K[1, :, n]
                                     + 2 * K[2, :, n]
                                     + K[3, :, n]) * deltaT / 6

        return x
