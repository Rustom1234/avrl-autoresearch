def equation(t, x, v, params):
    p=params
    return p[0]*np.cos(p[1]*t+p[2]) + p[3]*v**p[4] + p[5] + p[6]*v+p[7]*t