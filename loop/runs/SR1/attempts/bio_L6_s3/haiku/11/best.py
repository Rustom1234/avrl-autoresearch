import numpy as np
def equation(t, P, params):
    a,b,c,d,e,s=params[0],params[1],params[2],params[3],params[4],params[5]
    L=np.log(t+s)
    return a*P*L + b*P**2*L + c*P + d*P**2 + e*L + params[6]*P**3
