import numpy as np
import matplotlib.pyplot as plt
from integrator import *
from data import *
from settings import *


class KuramotoModel:

    def __init__(self, omega, K, eta, alpha, tau):

        self.N = omega.size
        self.omega = omega
        self.integr = Integration()
        self.data = Data()
        self.eta = eta
        self.K = K
        self.alpha = alpha
        self.tau = tau
        self.n = 1

    def __call__(self, theta, t):
        self.d_theta = np.zeros((self.N, len(t)))

        for i in range(self.N):
            self.d_theta[i, :] = self.omega[i] + (1 / self.N)\
                        * sum(self.K[i, j] * np.sin(theta[j -
                              int(self.tau[i, j]), :] - theta[i, :]
                              + self.alpha[i, j]) + self.eta[i, j] for j in
                              range(i - self.n, i + self.n))

        return self.d_theta

    def integrate(self, f, theta0, tf=100, integrator="RK4"):
        t = np.linspace(0, tf, 1000)

        if integrator == "RK4":
            theta = self.integr.RK4(self, theta0, t)
        elif integrator == "RK2":
            theta = self.integr.RK2(self, theta0, t)
        elif integrator == "Euler":
            theta = self.integr.euler(self, theta0, t)

        self.data.write_on_file(FILE['t'], t)
        self.data.write_on_file(FILE['theta'], theta % (2 * np.pi))

    def orders(self, theta):
        t = np.loadtxt(FILE['t'])
        z = np.zeros(self.N)
        R, phi = np.zeros(len(t)), np.zeros(len(t))
        for _t in range(len(t)):
            z = sum(np.exp(1j * theta[i][_t])
                    for i in range(self.N)) / self.N

            R[_t] = np.absolute(z)
            phi[_t] = np.angle(z)

        self.data.write_on_file(FILE['R'], R)
        self.data.write_on_file(FILE['phi'], phi)

    def shannon_entropy(self, theta):
        t = np.loadtxt(FILE['t'])
        n = self.n
        q = len(t)  # en attendant
        p = np.zeros(len(t))
        S = np.zeros(len(t))
        count = 0

        for i in range(len(t)):
            count += 1
            ib = (i - n) % self.N
            ia = (i + n) % self.N
            for a in range(q):
                inf = (2 * np.pi * a) / q
                sup = (2 * np.pi * (a + 1)) / q
                p[a] = sum(theta[k - 1] for k in range(ib, ia)
                           if inf <= theta[k - 1] < sup) / (2 * n + 1)

            S[i] = -sum(p[a] * np.log(p[a]) for a in range(q) if p[a] != 0)
            print(count)

        return S

    def shannon_entropies(self, theta):
        t = np.loadtxt(FILE['t'])
        S = np.zeros((len(t), self.N))
        for i in range(self.N):
            S[:, i] = self.shannon_entropy(theta[:, i])

        self.data.write_on_file(FILE['S'], S)
