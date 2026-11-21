def equation(t, x, v, params):
    p=params
    return p[0]*x+p[1]*x**3+p[2]*v+p[3]*np.exp(-p[4]*t)+p[5]+p[6]*x**p[7]/(v+p[8])+p[9]*v*np.abs(v)