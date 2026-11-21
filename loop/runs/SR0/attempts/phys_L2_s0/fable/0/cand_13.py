def equation(t, x, v, params):
    p=params
    return p[0]*x+p[1]*x*np.log(x)+p[2]/t+p[3]+p[4]*x**2/v+p[5]*x**p[6]+p[7]*v**p[8]+p[9]*x/v