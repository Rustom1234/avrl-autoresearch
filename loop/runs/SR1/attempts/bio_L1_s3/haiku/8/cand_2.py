import numpy as np
def equation(t, P, params):
    r, K, th = params[0], params[1], params[2]
    return r*P*(1-(P/K)**th)