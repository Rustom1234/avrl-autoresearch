import numpy as np
def equation(t, P, params):
    r, K, A, c = params[0], params[1], params[2], params[3]
    return r*P*(1-P/K)*(P/A-1) - c/P
