import numpy as np
def equation(t, P, params):
    a,b,c,d,e,f,g,h,k,m = params[:10]
    return P*(a+b*t+c*t*t) + d*P**2 + e*P**3 + f + g*t + h*t*t + k*t*P**2 + m*t*t*P**2