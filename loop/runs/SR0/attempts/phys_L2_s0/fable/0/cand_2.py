def equation(t, x, v, params):
    p=params
    return p[0]*x+p[1]*x**3+p[2]*v+p[3]*v*np.abs(v)+p[4]*t**p[5]+p[6]+p[7]*x*v