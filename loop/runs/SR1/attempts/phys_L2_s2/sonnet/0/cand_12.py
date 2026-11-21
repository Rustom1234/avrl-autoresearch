def equation(t, x, v, params):
    return params[0]*x/v + params[1]/x + params[2]*np.log(x) + params[3] + params[4]*x