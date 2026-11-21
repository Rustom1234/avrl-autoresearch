def equation(t, A, params):
    p=params
    return (p[0]+p[1]*t+p[2]*t**2) + (p[3]+p[4]*t+p[5]*t**2)/A + (p[6]+p[7]*t)*A + p[8]*A**2
