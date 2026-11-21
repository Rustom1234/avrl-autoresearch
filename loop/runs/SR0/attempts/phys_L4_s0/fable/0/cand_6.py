def equation(t, x, v, params):
    p=params
    return p[0]*np.sin(p[1]*x)+p[2]*x+p[3]*x*t+p[4]*t*(1-np.exp(-p[5]*v))+p[6]*v*v+p[7]*v+p[8]*np.cos(p[9]*t)