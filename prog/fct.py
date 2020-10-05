import numpy as np 
import matplotlib.pyplot as plt 

class Integration:
    """Integration by Euler's, RK2's, RK4's methods"""
    def euler(self, f, x, t):
        '''Approximate the solution of x' = f(x) by Euler's method.

        ---------- Args ----------
        
        f : function
            Right-hand side of the differential equation x' = f(x, t)
        x : ndarray
            Initial value x(t0) = x0 where t0 is the entry at index 0 in the array t
        t : ndarray
            1D NumPy array of t values where we approximate x values. Time step
            at each iteration is given by t[n+1] - t[n].

        ---------- Return ----------

        x : 1D NumPy array
            Approximation x[n] of the solution x(t_n) computed by Euler's method.
        '''

        for n in range(len(t)-1):
            deltaT = t[n + 1] - t[n]
            x[n + 1] = x[n] + f(x)[n] * deltaT

        return x

    def RK2(self, f, x, t):
        '''Approximate the solution of x' = f(x) by RK2's method.

        ---------- Args ----------
        
        f : function
            Right-hand side of the differential equation x' = f(x, t)
        x : ndarray
            Initial value x(t0) = x0 where t0 is the entry at index 0 in the array t
        t : ndarray
            1D NumPy array of t values where we approximate x values. Time step
            at each iteration is given by t[n+1] - t[n].

        ---------- Return ----------

        x : 1D NumPy array
            Approximation x[n] of the solution x(t_n) computed by RK2's method.
        '''

        for n in range(len(x)-1):
            deltaT = t[n + 1] - t[n]
            step = x + f(x)[n] * deltaT / 2
            x[n + 1] = x[n] + f(step)[n] * deltaT


        return x

    def RK4(self, f, x, t):
        '''Approximate the solution of x' = f(x) by RK4's method.

        ---------- Args ----------
        
        f : function
            Right-hand side of the differential equation x' = f(x, t)
        x : ndarray
            Initial value x(t0) = x0 where t0 is the entry at index 0 in the array t
        t : ndarray
            1D NumPy array of t values where we approximate x values. Time step
            at each iteration is given by t[n+1] - t[n].

        ---------- Return ----------

        x : 1D NumPy array
            Approximation x[n] of the solution x(t_n) computed by RK4's method.
        '''

        K = np.zeros((4, len(x) + 1))

        for n in range(len(K)+1):
            deltaT = t[n + 1] - t[n]
            K[0, n] = f(x)[n]
            K[1, n] = f(x + K[0, n] * deltaT / 2)[n]
            K[2, n] = f(x + K[1, n] * deltaT / 2)[n]
            K[3, n] = f(x + K[2, n] * deltaT)[n]

            x[n + 1] = x[n] + (K[0, n] + 2 * K[1, n] + 2 * K[2, n] + K[3, n])\
                            * deltaT / 6

        return x

class KuramotoModel:

    def __init__(self, omega):

        self.N = omega.size
        self.omega = omega
        self.d_theta = np.zeros(self.N)
        self.integr = Integration()

    def __call__(self, theta, t=0):
        for i in range(0, self.N):
            self.d_theta[i] = self.omega[i] + (1 / self.N)\
                         * sum(np.sin(theta[j] - theta[i]) for j in range(self.N))

        return self.d_theta

    def integrate(self, f, theta0, tf=100, integrator="RK4"):
        step = self.N+1
        t = np.linspace(0, tf + step / 2 , step-1)
        if integrator == "RK4":
            theta = self.integr.RK4(f, theta0, t)
        elif integrator == "RK2":
            theta = self.integr.RK2(f, theta0, t)
        elif integrator == "Euler":
            theta = self.integr.euler(f, theta0, t)

        return t, theta


