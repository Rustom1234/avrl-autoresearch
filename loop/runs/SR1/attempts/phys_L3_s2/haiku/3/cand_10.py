def equation(t, x, v, params):
    return params[0] + params[1]*v + params[2]*t + params[3]*np.log(x) + params[4]*np.cos(params[5]*t) + params[6]*x*v**2 + params[7]*x + params[8]*np.sin(params[5]*t)