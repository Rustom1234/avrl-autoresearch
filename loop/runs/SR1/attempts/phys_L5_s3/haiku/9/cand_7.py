def equation(t, x, v, params):
    w=params[2]
    return params[0]+params[1]*np.cos(w*t)+params[3]*np.sin(w*t)+params[4]*x+params[5]*x**2+params[6]*x**3+params[7]*v+params[8]*v*x