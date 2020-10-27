import numpy as np
from random import uniform, randint
from settings import *


class Data:
    """docstring for Data"""
    def __init__(self, N, M):
        self.omega = np.zeros(N)
        self.theta0 = np.zeros(N)
        self.K = np.zeros((N, N))
        self.eta = np.zeros((N, N))
        self.alpha = np.zeros((N, N))
        self.tau = np.zeros((N, N))
        self.M = M
        self.N = N

    def init_data(self, state="random"):

        if state == "random":
            self.random_states()
        elif state == "chimera":
            self.chimera_states()
        elif state == "inverse":
            self.inverse_states()
        elif state == "josephson":
            self.josephson_matrice()

        np.savetxt(FILE['omega'], self.omega)
        np.savetxt(FILE['theta0'], self.theta0)
        np.savetxt(FILE['K'], self.K)
        np.savetxt(FILE['eta'], self.eta)
        np.savetxt(FILE['alpha'], self.alpha)
        np.savetxt(FILE['tau'], self.tau)

    def write_on_file(self, file, wywtw):
        np.savetxt(file, wywtw)

    def random_states(self):
        for i in range(self.N):
            self.omega[i] = uniform(0, 3)
            self.theta0[i] = uniform(0, 2 * np.pi)
            for j in range(N):
                self.K[i, j] = uniform(0, 1e10)
                self.eta[i, j] = uniform(0, 0.5)
                self.alpha[i, j] = uniform(0, 2 * np.pi)
                self.tau[i, j] = randint(0, self.N // 2)

    def chimera_states(self):
        for i in range(self.N):
            self.omega[i] = 0.2 + i * 0.4 * np.sin(i**2 * np.pi / (2 * N**2))
            self.theta0[i] = uniform(0, 2 * np.pi)
            for j in range(N):
                if abs(i - j) <= self.M:
                    self.K[i, j] = uniform(0, 1e10)
                self.eta[i, j] = uniform(0, 0.5)
                self.alpha[i, j] = 1.46
                self.tau[i, j] = randint(0, self.N // 2)

    def inverse_states(self):
        for i in range(self.N):
            self.omega[i] = 0
            self.theta0[i] = uniform(0, 2 * np.pi)
            for j in range(N):
                if abs(i - j) <= self.M and abs(i - j) != 0:
                    self.K[i, j] = 1 / abs(i - j)
                else:
                    self.K[i, j] = 1e20
                self.eta[i, j] = uniform(0, 0.5)
                self.alpha[i, j] = 1.46
                self.tau[i, j] = abs(i - j)

    def josephson_matrice(self):
        for i in range(self.N):
            self.omega[i] = uniform(0, 3)
            self.theta0[i] = uniform(0, 2 * np.pi)
            for j in range(N):
                if abs(i - j) <= self.M:
                    self.K[i, j] = N * r * self.omega[i] *\
                                   (2 * np.exp(1) / hbar * r * Ib -
                                    self.omega[i] /
                                    np.sqrt((L * self.omega[i]**2
                                             - 1 / C)**2 + self.omega[i]**2 *
                                    (R + N * r)**2))
                self.eta[i, j] = uniform(0, 0.5)
                self.alpha[i, j] = np.arccos((L * self.omega[i]**2
                                             - 1 / C) /
                                             np.sqrt((L * self.omega[i]**2
                                                     - 1 / C)**2
                                                     + self.omega[i]**2 *
                                             (R + N * r)**2))
                self.tau[i, j] = randint(0, self.N // 2)
