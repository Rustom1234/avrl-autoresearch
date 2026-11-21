def equation(t, x, v, params):
    return params[0] + params[1]*t + params[2]*v + params[3]*t*np.exp(-params[4]*x) + params[5]*t**params[6]