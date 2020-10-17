import numpy as np
import networkx as nx
from kuramoto import *
from settings import *


class Graphs:
    """docstring for Graphs"""

    def __init__(self, kuramoto):
        self.kuramoto = kuramoto

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
        plt.figure()
        t = np.loadtxt(FILE['t'])
        title = "Kuramoto integrate by {0}'s method".format(integrator)
        xlabel = r"$t$"
        ylabel = r"$\theta$"
        self.graphs(t, theta, title, xlabel, ylabel, pol)

    def graph_density_kuramoto(self, N):
        plt.figure()
        t = np.loadtxt(FILE['t'])
        theta = np.loadtxt(FILE['theta'])
        ind = np.arange(N)

        plt.contourf(t, ind, theta)

        plt.xlabel("t")
        plt.ylabel("i")
        plt.colorbar()
        plt.show()

    def graph_orders(self):
        plt.figure()
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

    def graph_shannon_entropy(self, pol, t):
        plt.figure()
        S = np.loadtxt(FILE['S'])
        i = np.arange(len(S[t, :]))
        title = r"$i \longmapsto S_i^{q, n}(t)$"
        xlabel = r"$i$"
        ylabel = r"$S$"
        self.graphs(i, S[t, :], title, xlabel, ylabel, pol)

    def graph_density_shannon_entropy(self, N):
        plt.figure()
        t = np.loadtxt(FILE['t'])
        S = np.loadtxt(FILE['S'])
        ind = np.arange(N)

        plt.contourf(ind, t, S)

        plt.xlabel("i")
        plt.ylabel("t")
        plt.colorbar()
        plt.show()

    def graph_density_shannon_label(self, t, kuramoto):
        plt.figure()
        S = np.loadtxt(FILE['S'])
        r, c = np.zeros(N), np.zeros(N)
        for i in range(N):
            r[i], c[i] = kuramoto.label_to_coordinates(i)

        plt.contourf(r, c, S[:, t])

        plt.xlabel("r")
        plt.ylabel("c")
        plt.colorbar()
        plt.show()

    def graph_connectivity(self, kmin=0):
        K = np.loadtxt(FILE['K'])
        edges = self.kuramoto.connectivity(K, kmin)
        plt.figure()
        G = nx.Graph()
        G.add_edges_from(edges)
        nx.draw_networkx(G, with_labels=True)
        plt.show()
