def equation(t, x, v, params):
    return params[0]*np.cos(params[1]*t+params[2]) + params[3]*t*np.sin(params[4]*t+params[5]) + params[6]*x + params[7]*v + params[8]*v*v*t + params[9]*v*v
