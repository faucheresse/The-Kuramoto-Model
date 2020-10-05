import numpy as np 
import matplotlib.pyplot as plt 
from fct import *

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

#----------------------------------------------------------------------------------#
w = np.array([0.5, 0.5, 1.5, 0.5, 1.5, 0.5, 0.5, 0.5, 1.5, 1.5])
theta0 = np.array([0.91, 4.65, 0.94, 4.54, 1.14, 3.04, 0.57, 4.53, 6.15, 4.03])


kuramoto = KuramotoModel(w)
f  = kuramoto
t, theta = kuramoto.integrate(f, theta0, 1000000)


plt.plot(t, theta%(2 * np.pi), label="Kuramoto")
plt.legend()
plt.grid()
plt.show()
