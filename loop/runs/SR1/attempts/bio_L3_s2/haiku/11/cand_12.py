def equation(t, P, params):
    w=params[0]
    return (params[1] + params[2]*t + params[3]*P + params[4]*P*t + params[5]*P**2 + params[6]*t**2 + params[7]*np.sin(w*t)
            + params[8]*np.cos(w*t) + params[9]*P*np.sin(w*t))