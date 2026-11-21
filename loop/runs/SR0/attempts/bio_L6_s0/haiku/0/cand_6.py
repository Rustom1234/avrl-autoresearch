import numpy as np
def equation(t, P, params):
    a,b,c,d,e,f,g,h = params[:8]
    return (a + b*t)*P + (c + d*t)*P**2 + e*t**2 + f*t**3 + g*t + h
