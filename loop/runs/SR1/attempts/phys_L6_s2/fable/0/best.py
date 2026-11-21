def equation(t, x, v, params):
    return params[0]*np.sqrt(t) + params[1]*t**1.5/x + params[2]*v**2/(1+params[3]*v**2) + params[4] + params[5]*x + params[6]*v