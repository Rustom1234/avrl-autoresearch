def equation(t, x, v, params):
    return params[0]+params[1]*t**params[2]+params[3]*x+params[4]*np.exp(-x)+params[5]*x**2+params[6]*x**3+params[7]*v+params[8]*v*x