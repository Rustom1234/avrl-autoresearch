import numpy as np
def equation(t, A, params):
    k, n, Ae, m, c = params[0], params[1], params[2], params[3], params[4]
    return -k*A**n*(1-(A/Ae)**m) + c*t*A
