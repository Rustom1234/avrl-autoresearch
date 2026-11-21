import numpy as np
def equation(t,P,params):
    r,K,A,B,k,th=params[0],params[1],params[2],params[3],params[4],params[5]
    return r*(1-(P/K)**th)*(P-A)*(P-B)/P**k
