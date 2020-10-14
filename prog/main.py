import numpy as np
import matplotlib.pyplot as plt
from kuramoto import *
from graphs import *
from data import *
from settings import *


data = Data(N, M)
newData = False

if newData:
    data.init_data("chimera")

omega = np.loadtxt(FILE['omega'])
theta0 = np.loadtxt(FILE['theta0'])
K = np.loadtxt(FILE['K'])
eta = np.loadtxt(FILE['eta'])
alpha = np.loadtxt(FILE['alpha'])
tau = np.loadtxt(FILE['tau'])

kuramoto = KuramotoModel(omega, M, K, eta, alpha, tau)
graphs = Graphs()
f = kuramoto
# integrator = "Euler"
# integrator = "RK2"
integrator = "RK4"
tf = 100
# kuramoto.integrate(f, theta0, tf, integrator)

theta = np.loadtxt(FILE['theta'])
t = np.loadtxt(FILE['t'])

edges = kuramoto.connectivity(K)

# kuramoto.orders(theta)
# kuramoto.shannon_entropies(theta)

# -----graphs-----

# graphs.graph_kuramoto(theta[0], False, integrator)
# graphs.graph_kuramoto(theta[0], True, integrator)
# graphs.all_graph_kuramoto(theta)
# graphs.all_pol_graph_kuramoto(theta, 1)
# graphs.graph_density_kuramoto(omega.size)
# graphs.graph_orders()
# graphs.graph_shannon_entropy(False)
# graphs.graph_density_shannon_entropy(omega.size)
graphs.graph_connectivity(edges)
