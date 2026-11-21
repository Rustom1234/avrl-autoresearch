import numpy as np
def equation(t,P,params):
    a,h,K,q=params[0],params[1],params[2],params[3]
    return a*P/(h+P)*(1-(P/K)**q)
