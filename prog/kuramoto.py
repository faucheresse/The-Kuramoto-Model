import numpy as np 
import matplotlib.pyplot as plt
from settings import * 
from integrator import *

class KuramotoModel:

    def __init__(self, omega, kappa):

        self.N = omega.size
        self.omega = omega
        self.integr = Integration()
        self.kappa = kappa
        self.p = 1

    def __call__(self, theta, t=0):
        self.d_theta = np.zeros((len(theta), self.N))
        K = np.zeros((self.N, self.N))
        for i in range(len(K)):
            i = i % self.N
            for j in range(len(K)):
                if abs(i-j) <= self.p:
                    K[i, j] = self.kappa

        for i in range(self.N):
            self.d_theta[:][i] = self.omega[:][i] + (1 / self.N)\
                        * sum(K[i, j] * np.sin(theta[:][j] - theta[:][i])\
                                            for j in range(self.N))
        

        return self.d_theta

    def integrate(self, f, theta0, tf=100, integrator="RK4"):
        t = np.linspace(0, tf + self.N / 2, self.N)

        if integrator == "RK4":
            theta = self.integr.RK4(self, theta0, t)
        elif integrator == "RK2":
            theta = self.integr.RK2(self, theta0, t)
        elif integrator == "Euler":
            theta = self.integr.euler(self, theta0, t)

        return t, theta%(2 * np.pi)

    def graph_kuramoto(self, t, theta, integrator="RK4"):     
        plt.plot(t, theta, label="Kuramoto ({0})".format(integrator),\
                                                 marker="+")
        plt.xlabel(r"$t$")
        plt.ylabel(r"$\theta$")
        plt.title("Kuramoto integrate by %s"%integrator)
        plt.grid()
        plt.show()

    def r_exp_i_phi(self, theta):
        if theta.ndim == 1:
            z = sum(np.exp(theta*1j))/len(theta)
            return ([np.absolute(z), np.angle(z)])
        elif theta.ndim == 2:
            return np.array([self.r_exp_i_phi(theta[i])\
                        for i in range(np.shape(theta)[0])]).T

    def graph_r_i_phi(self, theta, t):
        R, phi = self.r_exp_i_phi(theta)
        plt.plot(t, R, label=r"$t \longmapsto R(t)$", marker="+")
        plt.plot(t, phi, label=r"$t \longmapsto \Phi(t)$", marker="+")

        plt.xlabel(r"$t$")
        plt.ylabel(r"$R, \Phi$")
        plt.legend()
        plt.grid()
        plt.show()

    def shanon_entropy(self, theta):
        p = np.zeros((self.N, self.N))
        n = (self.N - 1) // 2
        for i in range(self.N):
            count = [it for it in theta if (it == theta[:][i]).all()]
            p[:][i] = len(count) / len(theta)
        S = -sum(p[:][a] * np.log(p[:][a]) for a in range(self.N))
        print(p)

        return S

    def graph_shanon_entropy(self, t, theta):
        S = self.shanon_entropy(theta)
        plt.plot(t, S, label=r"$t \longmapsto S_i^{q, n}(t)$", marker="+")

        plt.xlabel(r"$t$")
        plt.ylabel(r"$S$")
        plt.legend()
        plt.grid()
        plt.show()

