def equation(t, x, v, params):
    w=params[0]
    return params[1]*np.sin(w*t+params[2])+params[3]*t+params[4]*t**2+params[5]
