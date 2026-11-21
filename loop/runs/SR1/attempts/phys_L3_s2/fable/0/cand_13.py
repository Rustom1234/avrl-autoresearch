def equation(t, x, v, params):
    p = params
    return p[0]*np.exp(-np.abs(p[1]*t)**(2.0 + p[2])) + p[3] + p[4]*v**1.5 + p[5]*x*np.sqrt(v)/(1 + np.abs(p[6])*x)