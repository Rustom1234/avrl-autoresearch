def equation(t, x, v, params):
    return params[0] + params[1]*x + params[2]*x**3 + v**params[3]*(params[4] + params[5]*t) + params[6]*np.cos(params[7]*t+params[8])