def equation(t, x, v, params):
    w=params[0]
    return params[1]*np.cos(w*t)+params[2]*np.sin(w*t)+params[3]+params[4]*x+params[5]*v+params[6]*x**3+params[7]*v*np.abs(v)
