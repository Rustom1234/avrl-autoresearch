def equation(t, x, v, params):
    p=params
    return p[0]*x+p[1]*v+p[2]*x**3+p[3]*x**p[4]*v**p[5]+p[6]*np.cos(p[7]*t+p[8])+p[9]