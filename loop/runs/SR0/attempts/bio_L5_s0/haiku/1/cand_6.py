import numpy as np
def equation(t, P, params):
    r0,r1,w,ph,K,c0,c1,w2,ph2 = params[0],params[1],params[2],params[3],params[4],params[5],params[6],params[7],params[8]
    return P*(r0+r1*np.sin(w*t+ph))*(1-P/K) + c0 + c1*np.sin(w2*t+ph2)
