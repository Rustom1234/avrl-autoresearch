def equation(t, x, v, params):
    return params[0]*x + params[1]*v + params[2]*x**3 + params[3]*np.sin(params[4]*t)