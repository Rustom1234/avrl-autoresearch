def equation(t, x, v, params):
    p=params
    return p[0]*x + p[1]*x**p[2] + p[3]*v + p[4]*v*x + p[6]*t**p[7] + p[8]*np.log(t) + p[9]