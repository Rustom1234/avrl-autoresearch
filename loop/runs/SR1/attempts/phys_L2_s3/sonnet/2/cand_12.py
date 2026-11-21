def equation(t, x, v, params):
    return params[0]+params[1]*x+params[2]*x**3+params[3]*v+params[4]*np.sqrt(t)+params[5]*np.sqrt(t)*x+params[6]*np.sin(params[7]*x)*t+params[8]*v*x