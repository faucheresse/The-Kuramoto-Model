import numpy as np
import matplotlib.pyplot as plt
from integrator import *
from data import *
from settings import *


class KuramotoModel:

    def __init__(self, omega, M, K, eta, alpha, tau):

        self.N = omega.size
        self.omega = omega
        self.integr = Integration()
        self.data = Data(N, M)
        self.eta = eta
        self.K = K
        self.alpha = alpha
        self.tau = tau
        self.n = M

    def __call__(self, theta, t):
        self.d_theta = np.zeros((self.N, len(t)))

        for i in range(self.N):
            self.d_theta[i, :] = self.omega[i] + (1 / self.N)\
                        * sum(self.K[i, j] * np.sin(theta[j -
                              int(self.tau[i, j]), :] - theta[i, :]
                              + self.alpha[i, j]) + self.eta[i, j] for j in
                              range((i - self.n) % self.N,
                                    (i + self.n) % self.N))

        return self.d_theta

    def coordinates_to_label(self, r, c):
        return r + c * Nr

    def label_to_coordinates(self, i):
        r = i % Nr
        c = (i - r) // Nr
        return r, c

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

    def shannon_entropy(self, theta, i):
        t = np.loadtxt(FILE['t'])
        n = self.n
        q = n
        p = np.zeros(q)
        S = np.zeros(len(t))
        count = 0

        for _t in range(len(t)):
            count += 1
            r, c = self.label_to_coordinates(i)
            rb, ra = (r - n), (r + n) % Nr
            cb, ca = (c - n), (c + n) % Nc
            for a in range(q):
                inf = (2 * np.pi * a) / q
                sup = (2 * np.pi * (a + 1)) / q
                for _c in range(cb, ca):
                    for _r in range(rb, ra):
                        norm = np.sqrt((_r - r)**2 + (_c - c)**2) <= n
                        j = self.coordinates_to_label(_r, _c) % self.N
                        if norm and (inf <= theta[j - 1] < sup):
                            p[a] += theta[j - 1] / ((2 * n + 1)**2 - 4)

            S[_t] = -sum(p[a] * np.log(p[a]) for a in range(q) if p[a] != 0)
            print(count)

        return S

    def shannon_entropies(self, theta):
        t = np.loadtxt(FILE['t'])
        S = np.zeros((len(t), self.N))
        for i in range(self.N):
            S[:, i] = self.shannon_entropy(theta[i, :], i)

        self.data.write_on_file(FILE['S'], S)

    def connectivity(self, K, kmin):
        edges = []
        for i in range(K.shape[0]):
            for j in range(K.shape[1]):
                if K[i, j] > kmin:
                    edges.append((i, j))

        return edges
