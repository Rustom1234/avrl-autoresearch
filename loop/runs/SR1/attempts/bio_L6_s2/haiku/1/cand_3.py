def equation(t, P, params):
    return -params[0]*P**params[1]*np.exp(params[2]*t)*(1+params[3]*t)
