import numpy as np
def equation(t, P, params):
    a0,a1,w,ph,b,c0,c1,w2,ph2 = params[0],params[1],params[2],params[3],params[4],params[5],params[6],params[7],params[8]
    return P*(a0+a1*np.sin(w*t+ph)) - b*P**2 + c0 + c1*np.sin(w2*t+ph2)
