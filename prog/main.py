import numpy as np 
import matplotlib.pyplot as plt 
from kuramoto import *
from graphs import *
from data import *
from settings import *

'''
Test d'intégration pour la fonction exponentielle
'''
# t = np.linspace(0, 2, 100)
# x0 = 1
# f = lambda x, t : x

# intgr = Integration()

# #---------- Test by Euler's method ----------#

# y1 = intgr.euler(f, x0, t)

# #---------- Test by RK2's method ----------#

# y2 = intgr.RK2(f, x0, t)

# #---------- Test by RK4's method ----------#

# y3 = intgr.RK4(f, x0, t)

# # plt.plot(t, y1, label="Euler")
# # plt.plot(t, y2, label="RK2")
# # plt.plot(t, y3, label="RK4")
# plt.plot(t, np.exp(t), label="exp")
# plt.legend()
# plt.grid()
# plt.show()

#------------------------------------------------------------#

data = Data()
newData = True
N = 100

if newData:
    data.init_data(N)

omega = np.loadtxt(FILE['omega'])
theta0 = np.loadtxt(FILE['theta0'])
K = np.loadtxt(FILE['K'])
eta = np.loadtxt(FILE['eta'])
alpha = np.loadtxt(FILE['alpha'])

kuramoto = KuramotoModel(omega, K, eta, alpha)
graphs = Graphs()
f = kuramoto
# integrator = "Euler"
# integrator = "RK2"
integrator = "RK4"
tf = 100
kuramoto.integrate(f, theta0, tf, integrator)

theta = np.loadtxt(FILE['theta'])
t = np.loadtxt(FILE['t'])

# kuramoto.orders(theta, t)
# kuramoto.shannon_entropies(theta, t)

#-----graphs-----#

# graphs.graph_kuramoto(theta[0], False, integrator)
# graphs.graph_kuramoto(theta[0], True, integrator)
# graphs.all_graph_kuramoto(theta)
# graphs.all_pol_graph_kuramoto(theta, 50)
# graphs.graph_density_kuramoto(omega.size, p = 20)
# graphs.graph_orders()
# graphs.graph_shannon_entropy(False)
# graphs.graph_density_shannon_entropy(omega.size, 50)

