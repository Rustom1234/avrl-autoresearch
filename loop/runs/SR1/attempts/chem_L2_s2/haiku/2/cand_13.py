def equation(t, A, params):
    p=params
    return -p[0]*(1-np.exp(-p[1]*A))*(1+p[2]*np.exp(-p[3]*t)) + p[4]*A + p[5]*A**2 + p[6]*A**3 + p[7]