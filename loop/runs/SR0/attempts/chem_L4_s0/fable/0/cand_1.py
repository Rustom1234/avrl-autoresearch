def equation(t, A, params):
    p=params
    return p[0]*(1-p[8]*np.exp(-p[1]*t)) + p[2]*A**p[3] + p[4]*A*np.exp(-p[5]*t) + p[6]*A*t+p[7]*A