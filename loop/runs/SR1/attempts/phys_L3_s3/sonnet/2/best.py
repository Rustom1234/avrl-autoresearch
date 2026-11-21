def equation(t, x, v, params):
    return params[0] + params[1]*t + params[2]*v + params[3]*x*v + params[4]*x + params[5]*np.sqrt(v) + params[6]*np.log(t)+params[7]*x**3