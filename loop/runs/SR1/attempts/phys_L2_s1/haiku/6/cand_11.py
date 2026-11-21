import numpy as np
def equation(t, x, v, params):
    F,w,ph,c,k,b,d,e,g,m0,h = params[0],params[1],params[2],params[3],params[4],params[5],params[6],params[7],params[8],params[9],params[0]*0
    return F*np.cos(w*t+ph) - c*v - k*x - b*x**3 - d*x**2 - e*v*v - g*x*v + m0 + h*np.exp(-x)
