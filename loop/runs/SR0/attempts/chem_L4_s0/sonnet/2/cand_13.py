def equation(t, A, params):
    return params[0]*(1-np.exp(-params[1]*A*t)) + params[2]*A*np.exp(-params[3]*t) *(1+params[4]*t)- params[5]*A*t*np.exp(-params[6]*t) - params[7]/A**2