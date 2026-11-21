def equation(t, A, params):
    p=params
    return -(p[0]+p[1]*np.exp(-p[2]*t))*A/(p[3]+A) + p[4]*(A-p[5])*A**p[6]/(1.0)