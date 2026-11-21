import numpy as np
def equation(t, x, v, params):
    A,w,phi,b,c,k,e,g,d,h = params[:10]
    return A*(1+b*v)*np.cos(w*t+phi) - c*v - k*x - e*x**3 - g*v*np.abs(v) - d*x**2 + h*x*v
