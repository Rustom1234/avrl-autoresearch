def equation(t, A, params):
    p=params
    return p[0]+p[1]*t**p[2]+p[3]*t**p[4]+p[5]*A**p[6]*t**p[7]