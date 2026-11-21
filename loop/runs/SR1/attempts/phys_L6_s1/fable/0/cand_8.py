def equation(t, x, v, params):
    return params[0]*np.cos(params[1]*t+params[2]) + params[3] + params[4]*v*t**params[5] + params[6]*x + params[7]*x**3 + params[8]*v