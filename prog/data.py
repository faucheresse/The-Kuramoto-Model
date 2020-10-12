import numpy as np
from random import uniform, randint
from settings import *

class Data:
    """docstring for Data"""

    def init_data(self, N):
        omega = np.zeros(N)
        theta0 = np.zeros(N)
        K = np.zeros((N, N))
        eta = np.zeros((N, N))
        alpha = np.zeros((N, N))
        tau = np.zeros((N, N))

        for i in range(N):
            omega[i] = uniform(0, 3)
            theta0[i] = uniform(0, 2 * np.pi)
            for j in range(N):
                K[i, j] = uniform(0, 2)
                eta[i, j] = uniform(0, 0.5)
                alpha[i, j] = uniform(0, 2 * np.pi)
                tau[i, j] = randint(0, N // 2)

        np.savetxt(FILE['omega'], omega)
        np.savetxt(FILE['theta0'], theta0)
        np.savetxt(FILE['K'], K)
        np.savetxt(FILE['eta'], eta)
        np.savetxt(FILE['alpha'], alpha)
        np.savetxt(FILE['tau'], tau)


    def write_on_file(self, file, wywtw):
        np.savetxt(file, wywtw)

