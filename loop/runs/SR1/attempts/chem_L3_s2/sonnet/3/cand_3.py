def equation(t, A, params):
    return params[0]*(1-np.exp(-params[1]*t))*np.exp(-params[2]*t) + params[3]*A**params[4]+params[5]*A*t