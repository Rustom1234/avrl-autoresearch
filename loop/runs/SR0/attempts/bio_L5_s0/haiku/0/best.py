import numpy as np
def equation(t, P, params):
    r,K,A,w,ph,B,C,D = params[0],params[1],params[2],params[3],params[4],params[5],params[6],params[7]
    return r*P*(1-P/K) + A*np.sin(w*t+ph) + B*t + C + D*P