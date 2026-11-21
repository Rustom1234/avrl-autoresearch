def equation(t, x, v, params):
    p=params
    return p[0]*t*np.exp(-p[1]*t) + p[2]/v + p[3] + p[4]*x