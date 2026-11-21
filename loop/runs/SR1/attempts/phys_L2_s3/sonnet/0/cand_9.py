def equation(t, x, v, params):
    return params[0]*x**3 + params[1]*t/x + params[2]*np.sqrt(t) + params[3]