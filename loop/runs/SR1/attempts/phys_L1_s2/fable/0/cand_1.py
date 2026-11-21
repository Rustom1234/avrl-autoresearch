def equation(t, x, v, params):
    return params[0]*x+params[1]*x**3+params[2]*v+params[3]*v*np.abs(v)+params[4]*np.cos(params[5]*t+params[6])+params[7]+params[8]*x**2