def equation(t, x, v, params):
    return params[0]*np.cos(params[1]*t)+params[2]*np.sin(params[1]*t)+params[3]*x+params[4]*x**3+params[5]*v+params[6]*v*np.abs(v)+params[7]*x*v+params[8]+params[9]*t