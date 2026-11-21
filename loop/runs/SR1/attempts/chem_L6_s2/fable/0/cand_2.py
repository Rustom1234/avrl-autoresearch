def equation(t, A, params):
    p=params
    return -p[0]*A**p[1]/(1+p[2]*A**p[3]) + (p[4]+p[5]*np.exp(p[6]*t))/A**p[7]