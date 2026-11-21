import numpy as np
def equation(t, P, params):
    a,b,c,d,e,g,h,k,m,n=params
    return a/P + b*P + c*P**2 + d*P**3 + e*t*P + g*t*P**2 + h*t/P + k*t + m*P*np.log(P) + n*t*P*np.log(P)
