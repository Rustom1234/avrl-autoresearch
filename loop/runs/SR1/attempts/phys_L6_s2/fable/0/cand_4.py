def equation(t, x, v, params):
    return params[0]*t*np.exp(params[1]*x) + params[2]*x + params[3]*np.abs(x)**params[4] + params[5]*v + params[6]*np.abs(v)**params[7] + params[8] + params[9]*t