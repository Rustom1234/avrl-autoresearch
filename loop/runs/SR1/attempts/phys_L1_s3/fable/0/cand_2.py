def equation(t, x, v, params):
    p=params
    return p[0]*x + p[1]*x**p[2] + p[3]*v + p[4]*v**p[5] + p[6]*np.sin(p[7]*t+p[8]) + p[9]