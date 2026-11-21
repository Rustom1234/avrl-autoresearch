import numpy as np
def equation(t, P, params):
    r0, r1, K, d, h = params[0], params[1], params[2], params[3], params[4]
    return P*(r0 + r1*t)*(1 - P/K) + d*t + h*P**2*0
