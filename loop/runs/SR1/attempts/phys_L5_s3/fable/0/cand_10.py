def equation(t, x, v, params):
    return params[0]*t**1.5+params[1]*x**params[2]*np.exp(-params[3]*x**params[4])+params[5]*x+params[6]*x**3+params[7]