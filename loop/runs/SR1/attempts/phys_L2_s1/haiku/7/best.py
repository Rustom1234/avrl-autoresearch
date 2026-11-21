def equation(t, x, v, params):
    return params[0] + params[1]*v + params[2]*x + params[3]*x**3 + params[4]*x*v + params[5]*x**2 + params[6]/v + params[7]*np.cos(params[8]*t) + params[9]*np.sin(params[8]*t)
