def equation(t, A, params):
    p=params
    return p[0] + p[1]*np.exp(-p[2]*t) + p[3]/A + p[4]*A*np.cos(p[5]*t+p[6]) + p[7]*A+p[8]*A**2