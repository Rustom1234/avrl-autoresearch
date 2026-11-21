import numpy as np
def equation(t, x, v, params):
    a,b,c,d,e,f,g,h,w,m = params[0],params[1],params[2],params[3],params[4],params[5],params[6],params[7],params[8],params[9]
    return a + b*np.cos(w*t) + c*np.sin(w*t) + d*x + e*v + f*x**3 + g*v*np.abs(v) + h*t + m*x*v
