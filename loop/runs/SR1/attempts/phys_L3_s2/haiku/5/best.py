def equation(t, x, v, params):
    w=params[0]
    return (params[1]*np.cos(w*t)+params[2]*np.sin(w*t)+params[3]
            + params[4]*x*np.exp(-params[5]*x) + params[6]*v*np.exp(-params[7]*v)
            + params[8]*x*v + params[9]*x*v*v)
