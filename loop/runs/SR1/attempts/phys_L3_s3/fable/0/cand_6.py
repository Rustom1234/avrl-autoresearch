def equation(t, x, v, params):
    p=params
    return p[0] + p[1]*t**p[2] + p[3]*x**p[4]*np.sin(p[5]*v+p[6]) + p[7]*v + p[8]*v**p[9]