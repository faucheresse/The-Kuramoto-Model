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
graphs = Graphs()
f = kuramoto

# ----- Computing -----

if newComputing:
    kuramoto.integrate(f, theta0, tf, integrator)
    kuramoto.orders()
    kuramoto.shannon_entropies()

# ----- graphs -----

theta = np.loadtxt(FILE['theta'])

graphs.kuramoto(theta[0], False, integrator)
graphs.kuramoto(theta[0], True, integrator)
graphs.dens_kuramoto(omega.size)
graphs.orders()
graphs.shannon(False, 0)
graphs.dens_shannon(omega.size)
graphs.graph_connectivity()
graphs.dens_shannon_coord(0)
graphs.dens_kuramoto_coord(0)
# graphs.anim_dens_shannon_coord(t)
# graphs.anim_dens_kuramoto_coord(t)
# graphs.anim_kuramoto(theta)
