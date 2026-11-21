def equation(t, A, params):
    p=params
    return p[0] + p[1]*np.exp(-p[2]*t) + p[3]*A**p[9] + p[4]*A*np.exp(-p[5]*t) + p[7]*A+p[8]*A*t+p[6]*t