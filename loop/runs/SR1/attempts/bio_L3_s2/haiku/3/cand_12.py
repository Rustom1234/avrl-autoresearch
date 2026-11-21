import numpy as np
def equation(t, P, params):
    w=params[0]
    s=np.sin(w*t); c=np.cos(w*t)
    s2=np.sin(2*w*t); c2=np.cos(2*w*t)
    return (params[1]*P + params[2]*P**2 + params[3] + params[4]/P
            + params[5]*s + params[6]*c + params[7]*s2 + params[8]*c2
            + P*(params[9]*s + params[10]*c + params[11]*s2 + params[12]*c2))
