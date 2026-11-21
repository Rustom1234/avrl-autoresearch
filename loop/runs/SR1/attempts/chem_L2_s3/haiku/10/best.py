def equation(t, A, params):
    p=params
    return -((p[0]+p[1]/t+p[2]*t) + A*(p[3]+p[4]/t+p[5]*t) + A**2*(p[6]+p[7]/t+p[8]*t))