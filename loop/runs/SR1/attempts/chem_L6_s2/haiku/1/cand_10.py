def equation(t, A, p):
    return -p[0]*A**p[1] + p[2]*np.exp(p[3]*t)/A**p[4] + p[5]