def equation(t, x, v, params):
    p=params
    return p[0] + p[1]*t*np.log(t) + p[2]*t + p[3]*np.sqrt(x)*np.sin(p[4]*v) + p[5]*v**p[6] + p[7]*v + p[8]*x