def equation(t, A, params):
    return params[0]*A**params[1] + params[2]*t/(1+params[3]*t) + params[4]*A**params[5]*t*np.exp(-params[6]*t)