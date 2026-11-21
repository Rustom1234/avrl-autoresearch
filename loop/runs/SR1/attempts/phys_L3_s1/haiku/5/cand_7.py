def equation(t, x, v, params):
    w=params[1]
    return params[0]*np.cos(w*t)+params[2]*np.sin(w*t) - params[3]*v - params[4]*v*np.abs(v) - params[5]*x - params[6]*x**3 + params[7]