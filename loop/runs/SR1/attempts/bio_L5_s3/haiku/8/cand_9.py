import numpy as np
def equation(t, P, params):
    r, A, K = params[0], params[1], params[2]
    return r*P*(P-A)*(1-P/K)
