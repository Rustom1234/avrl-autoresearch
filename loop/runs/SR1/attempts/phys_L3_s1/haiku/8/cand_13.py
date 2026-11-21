import numpy as np
def equation(t, x, v, params):
    A,w,phi,B,p2,c,k,e,g,d = params[:10]
    return A*np.cos(w*t+phi) + B*v*np.cos(w*t+p2) - c*v - k*x - e*x**3 - g*v*np.abs(v) - d*x**2
