def equation(t, x, v, params):
    return (params[0]+params[1]*np.sqrt(t)+params[2]*t+params[3]*np.sin(t)+params[4]*np.cos(t)
            +params[5]*x+params[6]*x**2+params[7]*x**3+params[8]*v+params[9]*v*np.abs(v))