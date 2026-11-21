import numpy as np
def equation(t, x, v, params):
    a0,a1,a2,a3,a4,a5,a6,a7,w,g = params[:10]
    h = params[9]*0 + g*0 + 0
    return (a0 + a1*x + a2*v + a3*x**3 + a4*x**2 + a5*x*v + a6*x**2*v + a7*np.abs(v)*v
            + g*np.cos(w*t) + params[8]*0)