import numpy as np
from random import uniform, randint
import time
from settings import *


class Data:
    """docstring for Data"""
    def __init__(self, N, M):
        self.omega = np.zeros(N)
        self.theta0 = np.zeros(N)
        self.K = np.zeros((N, N))
        self.eta = np.zeros((N, tf))
        self.alpha = np.zeros((N, N))
        self.tau = np.zeros((N, N))
        self.M = M
        self.N = N

    def init_data(self, state="random"):
        t1 = time.time()
        print("----- init data -----")

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
        t2 = time.time()
        print("running time : ", t2 - t1, "s")

    def write_on_file(self, file, wywtw):
        np.savetxt(file, wywtw)

    def random_states(self):
        for i in range(self.N):
            self.omega[i] = uniform(0, 3)
            self.theta0[i] = uniform(0, 2 * np.pi)
            for j in range(N):
                self.K[i, j] = uniform(0, 1e10)
                self.alpha[i, j] = uniform(0, 2 * np.pi)
                self.tau[i, j] = randint(0, self.N // 2)
            for _t in range(tf):
                self.eta[i, _t] = uniform(0, 0.5)

    def chimera_states(self):
        for i in range(self.N):
            self.omega[i] = 0.2 + i * 0.4 * np.sin(i**2 * np.pi / (2 * N**2))
            self.theta0[i] = uniform(0, 2 * np.pi)
            for j in range(N):
                if abs(i - j) <= self.M:
                    self.K[i, j] = uniform(0, 1e10)
                self.alpha[i, j] = 1.46
                self.tau[i, j] = randint(0, self.N // 2)
            for _t in range(tf):
                self.eta[i, _t] = uniform(0, 0.5)

    def inverse_states(self):
        for i in range(self.N):
            self.omega[i] = uniform(0, 3)
            self.theta0[i] = uniform(0, 2 * np.pi)
            for j in range(N):
                if abs(i - j) <= self.M and abs(i - j) != 0:
                    self.K[i, j] = 1 / abs(i - j)
                else:
                    self.K[i, j] = 1e20
                self.alpha[i, j] = 1.46
                self.tau[i, j] = abs(i - j)
            for _t in range(tf):
                self.eta[i, _t] = uniform(0, 0.5)

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
                self.alpha[i, j] = np.arccos((L * self.omega[i]**2
                                             - 1 / C) /
                                             np.sqrt((L * self.omega[i]**2
                                                     - 1 / C)**2
                                                     + self.omega[i]**2 *
                                             (R + N * r)**2))
                self.tau[i, j] = randint(0, self.N // 2)
            for _t in range(tf):
                self.eta[i, _t] = uniform(0, 0.5)
