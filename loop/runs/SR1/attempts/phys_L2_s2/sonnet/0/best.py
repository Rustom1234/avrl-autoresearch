def equation(t, x, v, params):
    return params[0]*x/v + params[1]*x**params[2] + params[3] + params[4]*x**params[5]+params[6]*np.cos(params[7]*t)