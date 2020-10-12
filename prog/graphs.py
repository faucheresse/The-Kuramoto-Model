import numpy as np
from kuramoto import *
from settings import *


class Graphs:
    """docstring for Graphs"""

    def graphs(self, x, y, title, xlabel, ylabel, pol):
        if not pol:
            ax = plt.subplot(111)
            ax.plot(x, y)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(True)

            plt.show()

        else:
            ax = plt.subplot(111, projection='polar')
            ax.plot(y, x)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(True)
            plt.show()

    def graph_kuramoto(self, theta, pol, integrator="RK4"):
        t = np.loadtxt(FILE['t'])
        title = "Kuramoto integrate by {0}'s method".format(integrator)
        xlabel = r"$t$"
        ylabel = r"$\theta$"
        self.graphs(t, theta, title, xlabel, ylabel, pol)

    def all_graph_kuramoto(self, theta):
        t = np.loadtxt(FILE['t'])
        for i in range(len(theta)):
            plt.plot(t, theta[i], label="Kuramoto ({0})".format(i))
        plt.xlabel(r"$t$")
        plt.ylabel(r"$\theta$")
        plt.title("Kuramoto")
        plt.grid()
        plt.show()

    def all_pol_graph_kuramoto(self, theta, p):
        t = np.loadtxt(FILE['t'])
        ax = plt.subplot(111, projection='polar')
        for i in range(0, len(theta), p):
            ax.plot(theta[i], t, label="Kuramoto ({0})".format(i))
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(r"$\theta$")
        ax.set_title("Kuramoto")
        ax.grid(True)
        plt.show()

    def graph_density_kuramoto(self, N):
        t = np.loadtxt(FILE['t'])
        theta = np.loadtxt(FILE['theta'])
        ind = np.arange(N)

        plt.contourf(t, ind, theta)

        plt.xlabel("t")
        plt.ylabel("i")
        plt.colorbar()
        plt.show()

    def graph_orders(self):
        t = np.loadtxt(FILE['t'])
        R = np.loadtxt(FILE['R'])
        phi = np.loadtxt(FILE['phi'])
        plt.plot(t, R, label=r"$t \longmapsto R(t)$")
        plt.plot(t, phi, label=r"$t \longmapsto \Phi(t)$")

        plt.xlabel(r"$t$")
        plt.ylabel(r"$R, \Phi$")
        plt.legend()
        plt.grid()
        plt.show()

    def graph_shannon_entropy(self, pol):
        t = np.loadtxt(FILE['t'])
        S = np.loadtxt(FILE['S'])
        i = np.arange(len(t))
        title = r"$i \longmapsto S_i^{q, n}(t)$"
        xlabel = r"$i$"
        ylabel = r"$S$"
        self.graphs(i, S[:, 0], title, xlabel, ylabel, pol)

    def graph_density_shannon_entropy(self, N):
        t = np.loadtxt(FILE['t'])
        S = np.loadtxt(FILE['S'])
        ind = np.arange(len(t))

        plt.contourf(t, ind, S)

        plt.xlabel("t")
        plt.ylabel("i")
        plt.colorbar()
        plt.show()
