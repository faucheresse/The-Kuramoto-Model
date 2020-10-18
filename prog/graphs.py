import numpy as np
import networkx as nx
from matplotlib.animation import ArtistAnimation
from PIL import Image
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

    def graph_kuramoto(self, theta, pol=False, integrator="RK4"):
        t = np.loadtxt(FILE['t'])
        title = "Kuramoto integrate by {0}'s method".format(integrator)
        xlabel = r"$t$"
        ylabel = r"$\theta$"
        self.graphs(t, theta, title, xlabel, ylabel, pol)

    def animated_kuramoto(self, theta):
        t = np.loadtxt(FILE['t'])
        for i in range(N):
            print(i)
            fig = plt.figure()
            ax = plt.subplot(111, projection='polar')
            ax.plot(t, theta[i])
            title = r"$\theta(t={0})$".format(i)
            xlabel = r"$t$"
            ylabel = r"$\theta$"
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.grid(True)
            fig.savefig("./animation/kuramoto{0}.png".format(i))
            plt.close(fig)

        im = [Image.open("./animation/kuramoto{0}.png".format(i))
                         for i in range(N)]

        im[0].save('./animation/kuramoto.gif',
                   format='GIF',
                   save_all=True,
                   append_images=im[1:],
                   duration=N//50,
                   loop=0)


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

    def graph_density_kuramoto_coordinates(self, t):
        plt.figure()
        theta = np.loadtxt(FILE['theta'])
        theta = np.reshape(theta[:, t], (Nc, Nr))
        r, c = np.arange(Nr), np.arange(Nc)

        plt.contourf(r, c, theta)

        plt.xlabel("r")
        plt.ylabel("c")
        plt.colorbar()
        # plt.show()
        plt.savefig("./animation/density_kuramoto{0}.png".format(t))
        plt.close()

    def animated_density_kuramoto_coordinates(self, t):
        plt.figure()
        theta = np.loadtxt(FILE['theta'])
        r, c = np.arange(Nr), np.arange(Nc)
        for i in range(len(t)):
            print(i)
            self.graph_density_kuramoto_coordinates(i)
        im = [Image.open("./animation/density_kuramoto{0}.png".format(i))
              for i in range(len(t))]

        im[0].save('./animation/density_kuramoto.gif',
                   format='GIF',
                   save_all=True,
                   append_images=im[1:],
                   duration=len(t)//50,
                   loop=0)

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

    def graph_density_shannon_coordinates(self, t):
        plt.figure()
        S = np.loadtxt(FILE['S'])
        S = np.reshape(S[t, :], (Nc, Nr))
        r, c = np.arange(Nr), np.arange(Nc)

        plt.contourf(r, c, S)

        plt.xlabel("r")
        plt.ylabel("c")
        plt.colorbar()
        # plt.show()
        plt.savefig("./animation/density_shannon{0}.png".format(t))
        plt.close()

    def animated_density_shannon_coordinates(self, t):
        plt.figure()
        S = np.loadtxt(FILE['S'])
        r, c = np.arange(Nr), np.arange(Nc)
        for i in range(len(t)):
            self.graph_density_shannon_coordinates(i)
        im = [Image.open("./animation/density_shannon{0}.png".format(i))
              for i in range(len(t))]

        im[0].save('./animation/density_shannon.gif',
                   format='GIF',
                   save_all=True,
                   append_images=im[1:],
                   duration=40,
                   loop=0)

    def graph_connectivity(self, kmin=0):
        K = np.loadtxt(FILE['K'])
        edges = self.kuramoto.connectivity(K, kmin)
        plt.figure()
        G = nx.Graph()
        G.add_edges_from(edges)
        nx.draw_networkx(G, with_labels=True)
        plt.show()
