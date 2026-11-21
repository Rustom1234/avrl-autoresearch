import numpy as np
def equation(t,P,params):
    r,K,A,B,e,w,ph=params[0],params[1],params[2],params[3],params[4],params[5],params[6]
    return r*(1+e*np.sin(w*t+ph))*(1-P/K)*(P-A)*(P-B)/P
