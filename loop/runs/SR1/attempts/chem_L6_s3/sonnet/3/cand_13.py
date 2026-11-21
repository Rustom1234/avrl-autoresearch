def equation(t, A, params):
    return params[0]*np.exp(-params[1]*t)*(1-np.exp(-params[2]*t**2)) - params[3]*A**params[4]*t**params[5]