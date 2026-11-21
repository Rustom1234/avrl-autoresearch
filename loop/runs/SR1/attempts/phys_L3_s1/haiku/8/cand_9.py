import numpy as np
def equation(t, x, v, params):
    A,w,phi,B,p2,D,p3,c,k,e = params[:10]
    return (A*np.cos(w*t+phi) + B*x*np.cos(w*t+p2) + D*v*np.cos(w*t+p3)
            - c*v - k*x - e*x**3)
