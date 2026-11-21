def equation(t, P, params):
    return params[0] + params[1]*np.sqrt(P)*(1+params[2]*np.cos(params[3]*t+params[4])) + params[5]*t + params[6]*P