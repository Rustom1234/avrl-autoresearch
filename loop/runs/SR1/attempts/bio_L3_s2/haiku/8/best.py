import numpy as np
def equation(t, P, params):
    a,b,c,d,e,f,w,ph,g,h = params
    return a + b*P + c*P**2 + d*t + g*t*P + (e + f*P)*np.sin(w*t + ph) + h*P**3
