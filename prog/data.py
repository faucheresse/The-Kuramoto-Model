import numpy as np
from random import uniform
from settings import *

class Data:
    """docstring for Data"""

    def init_data(self, N):
        omega = np.zeros(N)
        theta0 = np.zeros(N)

        for i in range(N):
            omega[i] = uniform(0, 3)
            theta0[i] = uniform(0, 2 * np.pi)

        np.savetxt(FILE['omega'], omega)
        np.savetxt(FILE['theta0'], theta0)
        np.savetxt(FILE['K'], K)


    def write_on_file(self, file, wywtw):
        np.savetxt(file, wywtw)

