import numpy as np
def equation(t, P, params):
    a,b,c,d,e,f,g,k = params[0],params[1],params[2],params[3],params[4],params[5],params[6],params[7]
    E = np.exp(k*t)
    return a + b*P + c*P**2 + d*P**3 + E*(e*P + f*P**2 + g*P**3)