def equation(t, x, v, params):
    return params[0]*x/v + params[1]/x + params[2]*np.log(1 + params[3]*x)