def equation(t, A, params):
    return params[0]*np.tanh(params[1]*t)*np.exp(-params[2]*t) - params[3]*t**params[4]*A**params[5]