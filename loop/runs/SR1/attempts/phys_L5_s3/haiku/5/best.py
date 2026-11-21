def equation(t, x, v, params):
    return params[0] + params[1]*x + params[2]*x**2 + params[3]*x**3 + params[4]*x**4 + params[5]*t + params[6]*t**2 + params[7]*v