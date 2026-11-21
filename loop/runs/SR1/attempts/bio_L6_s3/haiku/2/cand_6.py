import numpy as np
def equation(t,P,params):
    A,B,n,c,d,e=params[0],params[1],params[2],params[3],params[4],params[5]
    tn=np.abs(t)**n
    return -A*tn/(np.abs(B)**n+tn)+c*P+d*P**2+e*P*t
