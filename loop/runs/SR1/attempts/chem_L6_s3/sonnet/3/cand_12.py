def equation(t, A, params):
    return params[0]*np.exp(-params[1]*t)*np.tanh(params[2]*t**2) - params[3]*A**1.5*t**2*np.exp(params[4]*t)