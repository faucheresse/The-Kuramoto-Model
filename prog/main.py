import numpy as np
import matplotlib.pyplot as plt
from kuramoto import *
from graphs import *
from data import *
from settings import *


data = Data(N, M)

# ------ initialisation -----

if newData:
    data.init_data(state)

# ------ kuramoto -----
omega = np.loadtxt(FILE['omega'])
theta0 = np.loadtxt(FILE['theta0'])
K = np.loadtxt(FILE['K'])
eta = np.loadtxt(FILE['eta'])
alpha = np.loadtxt(FILE['alpha'])
tau = np.loadtxt(FILE['tau'])

kuramoto = KuramotoModel(omega, M, K, eta, alpha, tau)
graphs = Graphs(kuramoto)
f = kuramoto

# ----- Computing -----

if newComputing:
    kuramoto.integrate(f, theta0, tf, integrator)
    kuramoto.orders()
    kuramoto.shannon_entropies()

# -----graphs-----

theta = np.loadtxt(FILE['theta'])

# graphs.graph_kuramoto(theta[0], False, integrator)
# graphs.graph_kuramoto(theta[0], True, integrator)
# graphs.graph_density_kuramoto(omega.size)
# graphs.graph_orders()
graphs.graph_shannon_entropy(False, 0)
# graphs.graph_density_shannon_entropy(omega.size)
# graphs.graph_connectivity()
# graphs.graph_density_shannon_coordinates(0)
# graphs.graph_density_kuramoto_coordinates(0)
# graphs.animated_density_shannon_coordinates(t)
# graphs.animated_density_kuramoto_coordinates(t)
# graphs.animated_kuramoto(theta)
