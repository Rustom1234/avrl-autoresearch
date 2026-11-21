import numpy as np
def equation(t,P,params):
    r,K,a,b=params[0],params[1],params[2],params[3]
    return r*P*(1-P/K)*(1+a*P+b*P**2)
