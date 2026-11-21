def equation(t, A, params):
    return params[0]*np.exp(-params[1]*t)*np.tanh(params[5]*(t-params[6])) - params[2]*t**2*A**1.5