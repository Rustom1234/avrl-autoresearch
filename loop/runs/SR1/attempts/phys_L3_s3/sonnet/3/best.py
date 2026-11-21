def equation(t, x, v, params):
    return params[0]+params[1]*t+params[2]*np.sqrt(t)+params[3]*x**params[4]+params[5]*np.tanh(params[6]*v)+params[7]*v+params[8]*x*np.tanh(params[6]*v)+params[9]*x*v