def equation(t, A, params):
    return params[0] + params[1]*A + params[2]*A*np.exp(-params[3]*t) + params[4]*A**2