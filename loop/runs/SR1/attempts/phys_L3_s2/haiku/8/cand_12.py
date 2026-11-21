def equation(t, x, v, params):
    return params[0]*np.cos(params[1]*t+params[2]) - (params[3] + params[4]*x)*v - params[5]*np.abs(x)**params[6] - params[7]