def equation(t, A, params):
    return params[0]*np.exp(-params[1]*t)*(1+params[2]*np.sin(params[5]*t)) - params[3]*A**params[4]