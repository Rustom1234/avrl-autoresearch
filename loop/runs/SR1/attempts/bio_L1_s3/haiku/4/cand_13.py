import numpy as np
def equation(t,P,params):
    a,h,K,c,s=params[0],params[1],params[2],params[3],params[4]
    return a*P/(h+P)*(1-P/K)+c*np.log(1+P/s)
