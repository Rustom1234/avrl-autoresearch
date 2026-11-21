import numpy as np
def equation(t, x, v, params):
    a,b,c,d,e,f,w,g,h,k = params
    return (-a*x - b*x**3 - c*np.sin(d*x) - e*v - f*np.abs(v)*v + g*np.cos(w*t) + h*np.sin(w*t) + k)