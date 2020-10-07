import numpy as np 
import matplotlib.pyplot as plt
from settings import * 
from integrator import *
from scipy.stats import entropy

class KuramotoModel:

    def __init__(self, omega, kappa):

        self.N = omega.size
        self.omega = omega
        self.integr = Integration()
        self.kappa = kappa
        self.p = 1

    def __call__(self, theta):
        self.d_theta = np.zeros(self.N)
        K = np.zeros((self.N, self.N))
        for i in range(len(K)):
            i = i % self.N
            for j in range(len(K)):
                if abs(i-j) <= self.p:
                    K[i, j] = self.kappa

        for i in range(self.N):
            self.d_theta[i] = self.omega[i] + (1 / self.N)\
                        * sum(K[i, j] * np.sin(theta[j] - theta[i])\
                                                        for j in self.N)

        return self.d_theta

    def integrate(self, f, theta0, tf=100, h=0.01, integrator="RK4"):
        t = np.arange(0, tf + h / 2, h)

        if integrator == "RK4":
            theta = self.integr.RK4(f, theta0, t)
        elif integrator == "RK2":
            theta = self.integr.RK2(f, theta0, t)
        elif integrator == "Euler":
            theta = self.integr.euler(f, theta0, t)

        return t, theta

    def graph_kuramoto(self, t, theta, integrator="RK4"):     
        plt.plot(t, theta, label="Kuramoto ({0})".format(integrator),\
                                                 marker="+")
        plt.xlabel(r"$t$")
        plt.ylabel(r"$\theta$")
        plt.legend()
        plt.grid()
        plt.show()

    def r_exp_i_phi(self, theta):
        if theta.ndim == 1:
            z = sum(np.exp(theta*1j))/len(theta)
            return np.array([np.absolute(z), np.angle(z)])
        elif theta.ndim == 2:
            return np.array([r_exp_i_phi(theta[i]) for i in range(np.shape(theta)[0])]).T
        raise Exception('error')

    def R_exp_i_phi(self, f):
        R, phi = np.zeros(self.N), np.zeros(self.N)
        for i in range(1, self.N):
            _t, _theta = self.integrate(f, theta0, i)
            R[i - 1], phi[i - 1] = self.r_exp_i_phi(_theta)

        return R, phi%(2 * np.pi)

    def graph_r_i_phi(self, theta, t):
        R, phi = self.r_exp_i_phi(theta)
        plt.plot(t, R, label=r"$t \longmapsto R(t)$", marker="+")
        plt.plot(t, phi, label=r"$t \longmapsto \Phi(t)$", marker="+")

        plt.xlabel(r"$t$")
        plt.ylabel(r"$R, \Phi$")
        plt.legend()
        plt.grid()
        plt.show()

    def shanon_entropy(self, theta, n=2):
        p = np.zeros(self.N)
        for i in range(self.N):
            p[i] = theta[(i + n)%self.N] / theta[(i - n)%self.N]
        S = -sum(p[a] * np.log(p[a]) for a in range(self.N))

        return S

    def Shanon_entropy(self, f):
        S = np.zeros(self.N)
        S2 = np.zeros(self.N)
        for i in range(1, self.N):
            t, theta = self.integrate(f, theta0, i)
            S[i - 1] = self.shanon_entropy(theta)
            S2[i - 1] = entropy(theta)

        return S, S2

    def graph_shanon_entropy(self, f, t, theta):
        S, S2 = self.Shanon_entropy(f)
        plt.plot(t, S, label=r"$t \longmapsto S_i^{q, n}(t)$", marker="+")
        plt.plot(t, S2, label=r"$t \longmapsto S2_i^{q, n}(t)$", marker="+")

        plt.xlabel(r"$t$")
        plt.ylabel(r"$S$")
        plt.legend()
        plt.grid()
        plt.show()

