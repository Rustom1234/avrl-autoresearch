import numpy as np
def equation(t,P,params):
    a,r,K,c,d=params[0],params[1],params[2],params[3],params[4]
    return (a*np.exp(-c*t) + r*P*(1-P/K))*(1+d*t)
