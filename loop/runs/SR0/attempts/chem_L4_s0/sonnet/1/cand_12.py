def equation(t, A, params):
    return params[0]*np.exp(-params[1]/A)*(1-params[2]*np.exp(-params[3]*t)) - params[4] - params[5]*A*t + params[6]*np.sin(params[7]*t+params[8])