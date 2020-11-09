newData = True
newComputing = True

state = "chimera"
# state = "random"
# state = "inverse"
# state = "josephson"

# integrator = "Euler"
# integrator = "RK2"
integrator = "RK4"
tf = 100

Nr = 3
Nc = 4
N = Nr * Nc
M = N * 3 // 10

hbar = 6.626e-34

Ib = 1.5e-3
R = 50
L = 25e-12
C = 0.04e-12
r = 0.5

FILE = {'omega': "./parameters/omega.dat",
        'theta0': "./parameters/theta0.dat",
        'theta': "./parameters/theta.dat",
        't': "./parameters/t.dat",
        'R': "./parameters/R.dat",
        'phi': "./parameters/phi.dat",
        'S': "./parameters/S.dat",
        'K': "./parameters/K.dat",
        'eta': "./parameters/eta.dat",
        'alpha': "./parameters/alpha.dat",
        'tau': "./parameters/tau.dat"}
