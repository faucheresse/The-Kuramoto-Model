import numpy as np
from random import uniform, randint
from settings import *


class Data:
    """docstring for Data"""

    def init_data(self, N, M):
        omega = np.zeros(N)
        theta0 = np.zeros(N)
        K = np.zeros((N, N))
        eta = np.zeros((N, N))
        alpha = np.zeros((N, N))
        tau = np.zeros((N, N))

        # for i in range(N):
        #     omega[i] = uniform(0, 3)
        #     theta0[i] = uniform(0, 2 * np.pi)
        #     for j in range(N):
        #         K[i, j] = uniform(0, 2)
        #         eta[i, j] = uniform(0, 0.5)
        #         alpha[i, j] = uniform(0, 2 * np.pi)
        #         tau[i, j] = randint(0, N // 2)

# -----------chimère----------

        for i in range(N):
            omega[i] = 0.2 + i * 0.4 * np.sin(i**2 * np.pi / (2 * N**2))
            theta0[i] = uniform(0, 2 * np.pi)
            for j in range(N):
                if abs(i-j) <= M:
                    K[i, j] = 1
                eta[i, j] = uniform(0, 0.5)
                alpha[i, j] = 1.46
                tau[i, j] = randint(0, N // 2)

        np.savetxt(FILE['omega'], omega)
        np.savetxt(FILE['theta0'], theta0)
        np.savetxt(FILE['K'], K)
        np.savetxt(FILE['eta'], eta)
        np.savetxt(FILE['alpha'], alpha)
        np.savetxt(FILE['tau'], tau)

    def write_on_file(self, file, wywtw):
        np.savetxt(file, wywtw)
