def equation(t, x, v, params):
    return params[0]*x/v + params[1]*x**params[2] + params[3]*np.log(x) + params[4] + params[5]*np.cos(params[6]*t)+params[7]*np.sin(params[6]*t)