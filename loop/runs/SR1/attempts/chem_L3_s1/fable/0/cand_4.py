def equation(t, A, params):
    return params[0]*np.exp(-params[1]*t)/(1+params[5]*A**params[6]) - params[2]*A**params[3] + params[4]