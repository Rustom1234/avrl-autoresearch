def equation(t, x, v, params):
    return params[0]*x + params[1]/x + params[2]*v + params[3]*v**2 + params[4]*np.cos(params[5]*t+params[6]) + params[7]