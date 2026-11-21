import numpy as np
def equation(eps, T, params):
    a,b,c,d,f,g,h,k=params[:8]
    return (a+b*T)*eps**(c+d*T)*np.exp(-(f+g*T)*eps)+h+k*T
