import numpy as np
def equation(t,P,params):
    A,k,r,K,m,b,c=params[0],params[1],params[2],params[3],params[4],params[5],params[6]
    E=1-np.exp(-np.abs(k)*t)
    return -A*E + r*P*(1-P/np.abs(K))*np.exp(-np.abs(m)*t) + b*P*E + c*P**2*E
