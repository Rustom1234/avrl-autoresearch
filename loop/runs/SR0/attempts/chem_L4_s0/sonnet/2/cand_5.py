def equation(t, A, params):
    return params[0]*A**params[1]*(1-np.exp(-params[2]*t))*np.exp(-params[3]*t) + params[4] - params[5]/A**params[6]