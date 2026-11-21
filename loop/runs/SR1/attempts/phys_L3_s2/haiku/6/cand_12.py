import numpy as np
def equation(t, x, v, params):
    a0,a1,a2,a3,a4,a5,a6,a7,a8,w,a9 = params
    return (a0 + a1*x + a2*v + a3*x**3 + a4*x**2 + a5*x*v + a6*x**2*v + a7*np.abs(v)*v + a8*x/v
            + np.cos(w*t)*a9 + np.sin(w*t)*(a9*0+ (a9*0+1)*0))