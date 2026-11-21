def equation(t, x, v, params):
    w=params[0]
    return params[1]*np.cos(w*t)+params[2]*np.sin(w*t)+params[3]*v+params[4]*v*np.abs(v)+params[5]*x+params[6]*x**3+params[7]*x*v+params[8]+params[9]*x**2
