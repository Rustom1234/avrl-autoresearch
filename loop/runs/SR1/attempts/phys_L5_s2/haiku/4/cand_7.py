def equation(t, x, v, params):
    w=params[0]
    return params[1]*np.cos(w*t)+params[2]*np.sin(w*t)+params[3]+params[4]*np.cos(2*w*t)+params[5]*np.sin(2*w*t)
