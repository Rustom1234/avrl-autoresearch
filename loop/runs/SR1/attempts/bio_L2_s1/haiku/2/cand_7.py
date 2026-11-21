import numpy as np
def equation(t, P, params):
    a,b,c,d,e,f,g,h,k,m = params
    return a + b*np.log(t) + c*np.sqrt(t) - (d + e*t)*P**(f) * np.exp(-g*P) + h*t*P + k*np.log(t)*P**2 + m*P