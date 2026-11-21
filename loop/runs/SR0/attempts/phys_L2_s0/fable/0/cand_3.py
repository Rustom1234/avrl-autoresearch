def equation(t, x, v, params):
    p=params
    return p[0]*x+p[1]*x**3+p[2]*v+p[3]*v*np.abs(v)+p[4]*np.exp(-p[5]*t)*np.cos(p[8]*t+p[9])+p[6]+p[7]*x*v