def equation(t, x, v, params):
    p=params
    return p[0]*np.cos(p[1]*t+p[2]) + p[3]*x + p[4]*x**3 + p[5]*v**p[6] + p[7] + p[8]*np.sin(p[9]*x)