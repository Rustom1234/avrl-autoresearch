def equation(t, x, v, params):
    return params[0]+params[1]*t**params[2]+params[3]*x+params[4]*x**3+params[5]*np.log(x)+params[6]*x**params[7]