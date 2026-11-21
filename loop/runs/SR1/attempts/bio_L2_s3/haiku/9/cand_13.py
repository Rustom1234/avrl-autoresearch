import numpy as np
def equation(t, P, params):
    a,b,c,d,e,f = params[:6]
    return a + b*t + c*P + d*P**2 + e*P**3 + f*P*t