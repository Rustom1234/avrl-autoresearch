def equation(t, x, v, params):
    return params[0] + params[1]*x + params[2]/x + params[3]*np.sqrt(x) + params[4]*np.exp(-x) + params[5]*x/v**params[6] + params[7]*x**params[8] + params[9]*x**3