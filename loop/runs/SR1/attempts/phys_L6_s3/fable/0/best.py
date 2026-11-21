def equation(t, x, v, params):
    p=params
    return p[0]*np.sqrt(t)/v+p[1]*np.sqrt(x)+p[2]*np.cos(p[3]*t**p[4]+p[5])+p[6]+p[7]*v+p[8]*x