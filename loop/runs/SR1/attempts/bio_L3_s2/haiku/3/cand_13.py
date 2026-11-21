import numpy as np
def equation(t, P, params):
    w=params[0]
    s=np.sin(w*t); c=np.cos(w*t)
    s2=np.sin(2*w*t); c2=np.cos(2*w*t)
    return (params[1]*P + params[2] + params[3]/P
            + params[4]*s + params[5]*c + params[6]*s2 + params[7]*c2
            + P*(params[8]*s2 + params[9]*c2))
