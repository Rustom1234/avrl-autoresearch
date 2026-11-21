def equation(t, x, v, params):
    return params[0]*np.cos(params[1]*t+params[2]) + params[3] + params[4]*v + params[5]*x + params[6]*x**3 + params[7]*v*np.abs(v) + params[8]*x*v