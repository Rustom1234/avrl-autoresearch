import numpy as np
def equation(t,P,params):
    r,K,A,B,k=params[0],params[1],params[2],params[3],params[4]
    return r*(1-P/K)*(P-A)*(P-B)/P**k
