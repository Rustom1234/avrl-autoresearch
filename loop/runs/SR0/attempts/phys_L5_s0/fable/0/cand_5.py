def equation(t, x, v, params):
    p=params
    return p[0]*t**p[1]*np.exp(-p[2]*t) + p[3]*v**p[4] + p[5] + p[6]*v