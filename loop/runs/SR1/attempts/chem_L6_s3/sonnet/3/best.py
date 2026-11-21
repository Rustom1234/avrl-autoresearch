def equation(t, A, params):
    return params[0]*np.exp(-params[1]*t)*np.tanh(params[2]*t**params[3]) - params[4]*A**params[5]*t**params[6]*np.exp(params[7]*t)