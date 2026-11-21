def equation(t, x, v, params):
    p=params
    return p[0]*x+p[1]*np.tanh(p[2]*x)+p[3]*t**p[5]+p[4]*np.log(t)+p[6]+p[7]*x**p[8]/v**p[9]