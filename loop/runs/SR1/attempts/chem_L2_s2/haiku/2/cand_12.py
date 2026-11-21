def equation(t, A, params):
    p=params
    P=p[0]+p[1]*np.exp(-p[2]*t)
    return -P*A/(p[3]+A) + p[4]*A + p[5]*A**2 + p[6]*A**3 + p[7]