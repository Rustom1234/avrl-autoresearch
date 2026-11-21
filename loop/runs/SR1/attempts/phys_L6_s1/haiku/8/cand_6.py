def equation(t, x, v, params):
    return params[0]*np.cos(params[1]*t+params[2]) + params[3]*x + params[4]*v + params[5]*x**3 + params[6]*v**2 + params[7]*x*v + params[8]*t + params[9]
