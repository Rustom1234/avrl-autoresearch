def equation(t, x, v, params):
    return params[0]*np.exp(-params[1]*t)+params[2]+params[3]*np.abs(v)**params[4]+params[5]*v+params[6]*x+params[7]*x**3