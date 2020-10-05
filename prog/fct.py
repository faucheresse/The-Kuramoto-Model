import numpy as np 
import matplotlib.pyplot as plt 

class Integration:
    """docstring for Integration"""
    def euler(self, f, x0, t):
        '''Approximate the solution of x' = f(x,t) by Euler's method.

        ---------- Args ----------
        
        f : function
            Right-hand side of the differential equation x' = f(t, x), x(t0) = x0
        x0 : number
            Initial value x(t0) = x0 where t0 is the entry at index 0 in the array t
        t : array
            1D NumPy array of t values where we approximate y values. Time step
            at each iteration is given by t[n+1] - t[n].

        ---------- Return ----------

        x : 1D NumPy array
            Approximation x[n] of the solution x(t_n) computed by Euler's method.
        '''

        x = np.zeros(len(t))
        x[0] = x0

        for n in range(len(t)-1):
            deltaT = t[n + 1] - t[n]
            x[n + 1] = x[n] + f(x, t)[n] * deltaT

        return x

    def RK2(self, f, x0, t):
        '''Approximate the solution of x' = F(x,t) by RK2's method.

        ---------- Args ----------
        
        f : function
            Right-hand side of the differential equation x' = f(t, x), x(t0) = x0
        x0 : number
            Initial value x(t0) = x0 where t0 is the entry at index 0 in the array t
        t : array
            1D NumPy array of t values where we approximate y values. Time step
            at each iteration is given by t[n+1] - t[n].

        ---------- Return ----------

        x : 1D NumPy array
            Approximation x[n] of the solution x(t_n) computed by RK2's method.
        '''

        x = np.zeros(len(t))
        x[0] = x0

        for n in range(len(t)-1):
            deltaT = t[n + 1] - t[n]
            step = x + f(x, t)[n] * deltaT / 2
            x[n + 1] = x[n] + f(step, t)[n] * deltaT


        return x

    def RK4(self, f, x, t):
        '''Approximate the solution of x' = F(x,t) by RK4's method.

        ---------- Args ----------
        
        f : function
            Right-hand side of the differential equation x' = f(x, t), x(t0) = x0
        x0 : number
            Initial value x(t0) = x0 where t0 is the entry at index 0 in the array t
        t : array
            1D NumPy array of t values where we approximate y values. Time step
            at each iteration is given by t[n+1] - t[n].

        ---------- Return ----------

        x : 1D NumPy array
            Approximation x[n] of the solution x(t_n) computed by RK2's method.
        '''

        # x = np.zeros(len(t))
        # x[0] = x0

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

    def integrate(self, f, theta0, tf=100):
        step = self.N+1
        t = np.linspace(0, tf + step / 2 , step-1)
        theta = self.integr.RK4(f, theta0, t)

        return t, theta


