import numpy as np
def equation(t, P, params):
    r0, r1, K, a, b = params[0], params[1], params[2], params[3], params[4]
    return (r0 + r1*t)*P*(1-P/K) - a/P + b*t
