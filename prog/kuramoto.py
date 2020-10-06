import numpy as np 
import matplotlib.pyplot as plt
from settings import * 
from integrator import *

class KuramotoModel:

    def __init__(self, omega):

        self.N = omega.size
        self.omega = omega
        self.integr = Integration()

    def __call__(self, theta, t=0):
        self.d_theta = np.zeros(self.N)
        for i in range(self.N):
            self.d_theta[i] = self.omega[i] + (1 / self.N)\
                         * sum(np.sin(theta[j] - theta[i])\
                                     for j in range(self.N))

        return self.d_theta

    def integrate(self, f, theta0, tf=100, integrator="RK4"):
        step = self.N + 1
        t = np.linspace(0, tf + step / 2 , step-1)

        if integrator == "RK4":
            theta = self.integr.RK4(f, theta0, t)
        elif integrator == "RK2":
            theta = self.integr.RK2(f, theta0, t)
        elif integrator == "Euler":
            theta = self.integr.euler(f, theta0, t)

        return t, theta

    def r_exp_i_phi(self, theta):
        i = complex(0, 1)
        z = (1 / self.N) * sum(np.exp(i * theta[j])\
                        for j in range(self.N))

        return z.real, z.imag

    def R_exp_i_phi(self, f, theta):
        R, phi = np.zeros(self.N), np.zeros(self.N)
        for i in range(1, self.N):
            _t, _theta = self.integrate(f, theta0, i)
            R[i-1], phi[i-1] = self.r_exp_i_phi(theta)

        return R, phi

    def shanon_entropy():
        pass

