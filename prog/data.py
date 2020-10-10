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


        with open(theta0File, "w+", encoding='utf-8') as f:
            for i in range(N):
                f.write(str(uniform(0, 2 * np.pi)) + "\n")


    def write_on_file(self, file, wywtw):
        np.savetxt(file, wywtw)

