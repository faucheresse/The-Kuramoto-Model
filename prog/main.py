import numpy as np 
import matplotlib.pyplot as plt 
from kuramoto import *
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
kuramoto = KuramotoModel(omega, 1)
f = kuramoto
integrator = "RK4"
t, theta = kuramoto.integrate(f, theta0, 100, integrator)
    

# kuramoto.graph_kuramoto(t, theta, integrator)
# kuramoto.graph_r_i_phi(theta, t)
# kuramoto.graph_shanon_entropy(t, theta)

