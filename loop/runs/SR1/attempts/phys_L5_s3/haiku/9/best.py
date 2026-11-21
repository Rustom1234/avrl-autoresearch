def equation(t, x, v, params):
    return params[0]+params[1]*t**1.5+params[2]*x+params[3]*x**2+params[4]*x**3+params[5]*np.exp(-params[6]*x)+params[7]*x*np.exp(-x)