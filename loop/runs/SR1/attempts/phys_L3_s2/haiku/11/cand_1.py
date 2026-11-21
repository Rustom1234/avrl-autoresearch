def equation(t, x, v, params):
    return params[0]*x + params[1]*v + params[2]*np.cos(params[3]*t) + params[4]*x**3 + params[5]*v*np.abs(v)