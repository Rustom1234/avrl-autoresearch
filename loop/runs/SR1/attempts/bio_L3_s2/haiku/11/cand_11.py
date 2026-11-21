def equation(t, P, params):
    w=params[0]
    return (params[1] + params[2]*t + params[3]*P + params[4]*P*t + params[5]*P**2 + params[10]*t**2 + params[11]*P*t**2
            + (params[6] + params[7]*P)*np.sin(w*t) + (params[8] + params[9]*P)*np.cos(w*t))