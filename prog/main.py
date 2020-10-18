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
graphs = Graphs(kuramoto)
f = kuramoto
# integrator = "Euler"
# integrator = "RK2"
integrator = "RK4"
tf = 100
# kuramoto.integrate(f, theta0, tf, integrator)

theta = np.loadtxt(FILE['theta'])
t = np.loadtxt(FILE['t'])

# kuramoto.orders(theta)
# kuramoto.shannon_entropies(theta)

# -----graphs-----

# graphs.graph_kuramoto(theta[29], False, integrator)
# graphs.graph_kuramoto(theta[29], True, integrator)
# graphs.graph_density_kuramoto(omega.size)
# graphs.graph_orders()
# graphs.graph_shannon_entropy(False, 0)
# graphs.graph_density_shannon_entropy(omega.size)
# graphs.graph_connectivity()
# graphs.graph_density_shannon_coordinates(0)
# graphs.graph_density_kuramoto_coordinates(50)
# graphs.animated_density_shannon_coordinates(t)
graphs.animated_density_kuramoto_coordinates(t)
graphs.animated_kuramoto(theta)
