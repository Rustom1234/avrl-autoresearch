def equation(t, x, v, params):
    return params[0]*x**params[1]*v**params[2] + params[3]*x + params[4]*v + params[5]*x**3 + params[6]/v + params[7]*np.cos(params[8]*t) + params[9]
