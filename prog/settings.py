import numpy as np
from random import uniform

N = 10
omegaFile = "./parameters/omega.dat"
theta0File = "./parameters/theta0.dat"

with open(omegaFile, "w+") as f:
    
    for i in range(N):
        s = ''
        s += str(uniform(0, 3))
        s += '\n'
        f.write(s)

with open(theta0File, "w+") as f:
    
    for i in range(N):
        s = ''
        s += str(uniform(0, 2 * np.pi))
        s += '\n'
        f.write(s)

omega = np.loadtxt(omegaFile)
theta0 = np.loadtxt(theta0File)
