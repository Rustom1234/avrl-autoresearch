def equation(t, x, v, params):
    return (params[0] + params[1]*t + params[2]*t**2 + params[3]*t**3 + params[4]*x + params[5]*t*x
            + params[6]*np.sin(x) + params[7]*np.sin(v) + params[8]*v + params[9]*v*np.abs(v)
            + params[1]*0*x*v)