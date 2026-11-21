def equation(t, x, v, params):
    p=params
    return p[0]*x+p[1]*v+p[2]*x**3+p[3]*v**p[5]+p[4]*x**p[6]*np.log(v)+p[7]*np.cos(p[8]*t)+p[9]