def equation(t, x, v, params):
    p = params
    return p[0]*np.sin(p[1]*t) + p[2]*np.cos(p[1]*t) + p[3] + p[4]*v**p[5] + p[6]*x*v**p[7]/(1 + p[8]*x) + p[9]*x