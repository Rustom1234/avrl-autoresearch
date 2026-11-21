def equation(t, A, params):
    return params[0]*A**2*np.exp(params[1]*t) + params[2]*A**params[3]*t**params[4] + params[5]*A
