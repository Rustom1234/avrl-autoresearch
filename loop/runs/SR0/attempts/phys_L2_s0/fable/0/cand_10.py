def equation(t, x, v, params):
    p=params
    return p[0]*x+p[1]*x**p[2]+p[3]*(t+p[4])**p[5]+p[6]+p[7]*x**p[8]/v**p[9]