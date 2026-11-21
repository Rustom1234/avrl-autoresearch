def equation(t, P, params):
    w = params[7]
    return params[0] + params[1]*P + params[2]*P**2 + params[3]*t*P + params[4]*t + params[5]*np.sin(w*t) + params[6]*np.cos(w*t) + params[8]*t*P**2