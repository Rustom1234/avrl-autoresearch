def equation(t, x, v, params):
    return params[0]*np.sqrt(t) + params[1]*np.tanh(params[2]*x) + params[3]*v**2*np.abs(x)**params[4]