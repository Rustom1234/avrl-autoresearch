def equation(t, x, v, params):
    return params[0]*np.cos(params[1]*t+params[2]) + params[3]*np.sin(params[4]*t+params[5]) + params[6]*v*v*t + params[7]*x + params[8]*v + params[9]
