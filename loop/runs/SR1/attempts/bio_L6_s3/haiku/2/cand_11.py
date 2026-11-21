import numpy as np
def equation(t,P,params):
    r,K,A,k,c,d=params[0],params[1],params[2],params[3],params[4],params[5]
    E=1-np.exp(-np.abs(k)*t)
    return r*P*(1-P/K)-A*E+c*P*E+d*P**2*E
