import numpy as np
def equation(t, A, params):
    k,n,K,a,b,c = params[0],params[1],params[2],params[3],params[4],params[5]
    return k*A**n/(1+K*A)*t**a*np.exp(-b*t)*np.exp(c*0)
