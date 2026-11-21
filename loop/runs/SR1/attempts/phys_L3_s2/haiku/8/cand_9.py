def equation(t, x, v, params):
    return params[0]*np.cos(params[1]*t+params[2]) - params[3]*np.abs(v)**params[4]*np.sign(v) - params[5]*np.abs(x)**params[6]*np.sign(x) - params[7]