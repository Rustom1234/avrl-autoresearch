import numpy as np
def equation(t, P, params):
    a,b,c,d,e,f,g,w,ph,h = params[0],params[1],params[2],params[3],params[4],params[5],params[6],params[7],params[8],params[9]
    return (a + b*t + g*np.sin(w*t+ph))*P + c*P**2 + d + e*t + f/P + h*np.cos(w*t+ph)
