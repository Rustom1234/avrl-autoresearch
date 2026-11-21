import numpy as np
def equation(t, P, params):
    r, K, h, A, c = params[0], params[1], params[2], params[3], params[4]
    return r*P*(1-P/K) - h*P/(A+P) + c
