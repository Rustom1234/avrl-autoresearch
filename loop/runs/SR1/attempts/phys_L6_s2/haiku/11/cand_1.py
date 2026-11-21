def equation(t, x, v, params):
    return params[0]*np.cos(params[1]*t) - params[2]*v - params[3]*x - params[4]*x**3