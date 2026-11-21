def equation(t, x, v, params):
    return params[0]+params[1]*np.tanh(params[2]*v)+params[3]*x+params[4]*x**2+params[5]*t+params[6]*t**2+params[7]*v*x+params[8]*v+params[9]*np.sqrt(t)