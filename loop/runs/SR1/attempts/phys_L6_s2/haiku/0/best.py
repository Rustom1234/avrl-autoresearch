import numpy as np
def equation(t, x, v, params):
    a,b,c,d,e,f,g,h,k,m = params[0],params[1],params[2],params[3],params[4],params[5],params[6],params[7],params[8],params[9]
    E = np.exp(e*t)
    return a + b*x + c*v + d*x*v + f*x**2 + g*v**2 + E*(h + k*x + m*v)
