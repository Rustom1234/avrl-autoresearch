import numpy as np
def equation(t, P, params):
    a,b,g,c,e,d,w,ph,f,h = params
    return a + b*P**g + c*(1-np.exp(-e*t)) + d*np.sin(w*t+ph) + f*t + h*P*np.sin(w*t+ph)
