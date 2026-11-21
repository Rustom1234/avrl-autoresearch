def equation(t, x, v, params):
    return params[0] + params[1]*t + params[2]*t**2 + params[3]*v + params[4]*v*np.abs(v) + params[5]*v**3 + params[6]*x + params[7]*x*v + params[8]*x**3 + params[9]*t*x