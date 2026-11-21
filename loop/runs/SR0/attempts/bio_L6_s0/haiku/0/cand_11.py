import numpy as np
def equation(t, P, params):
    a,b,c,d,e,f = params[:6]
    r = a + b*t
    K = c + d*t
    return r*P*(1 - P/K) + e + f*t
