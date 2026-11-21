def equation(t, A, params):
    return params[0]*(1-np.exp(-params[1]*t))*np.tanh(params[2]*A) - params[3]*A*t - params[4]/A**params[5]+params[6]