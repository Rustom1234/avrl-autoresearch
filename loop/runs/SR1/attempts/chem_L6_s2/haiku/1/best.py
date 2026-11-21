def equation(t, A, p):
    g = p[0] + p[1]*np.log(A) + p[2]*A + p[3]*A**2
    return g*np.exp(p[4]*t) + p[5] + p[6]*t + p[7]*t*np.log(A) + p[8]*t/A