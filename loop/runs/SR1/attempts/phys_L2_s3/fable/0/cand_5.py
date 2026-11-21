def equation(t, x, v, params):
    p=params
    return p[0]*np.sin(p[1]*t+p[2])+p[3]*x+p[4]*x**3+p[5]*v+p[6]*t/x+p[7]/x+p[8]+p[9]*v*np.abs(v)