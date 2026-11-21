def equation(t, A, params):
    return params[0] + params[1]*A + params[2]*A**2 + params[3]*t**2 + (params[4] + params[5]*t**3)/A + params[6]*np.exp(-t)/A + params[7]*t