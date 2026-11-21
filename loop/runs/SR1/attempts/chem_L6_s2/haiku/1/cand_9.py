def equation(t, A, p):
    return -p[0]*A**p[1] + p[2]*A**p[3]*np.exp(p[4]*t) + p[5]