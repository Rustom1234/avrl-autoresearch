import numpy as np
def equation(t, P, params):
    H, tau, K, a, Kg, b = params[0], params[1], params[2], params[3], params[4], params[5]
    return -H*np.tanh(t/tau)*(1-P/K)*(1+b*P) + a*P*(1-P/Kg)
