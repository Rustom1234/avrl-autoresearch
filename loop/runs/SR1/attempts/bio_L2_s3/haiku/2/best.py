import numpy as np
def equation(t, P, params):
    p=params
    return p[0] + p[1]*t + P*(p[2]+p[3]*t) + P**2*(p[4]+p[5]*t+p[6]*t**2) + P**3*p[7] + t**2*p[8] + t**2*P*p[9]
