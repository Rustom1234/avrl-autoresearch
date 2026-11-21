import numpy as np
def equation(t, P, params):
    a,b,c,pw,q,r_,d,e = params[:8]
    return a*P*t**pw + b*P**2*t**q + c*t**r_ + d*P + e*P**1.0
