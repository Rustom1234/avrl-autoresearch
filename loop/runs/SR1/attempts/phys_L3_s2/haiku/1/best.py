import numpy as np
def equation(t, x, v, params):
    c0,k,cv,a3,F,G,w,cq,d,e = params
    return (c0 - k*x - cv*v - a3*x**3 + F*np.cos(w*t) + G*np.sin(w*t)
            - cq*v*np.abs(v) + d*x*v + e*t*x)
