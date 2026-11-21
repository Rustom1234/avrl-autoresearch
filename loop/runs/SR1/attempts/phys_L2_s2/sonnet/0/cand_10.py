def equation(t, x, v, params):
    return params[0]*x/v + params[1]/x + params[2]*x**params[3] + params[4] + params[5]*np.cos(params[6]*t)