def equation(t, x, v, params):
    p=params
    return p[0]*x+p[1]*x*np.log(x)+p[2]/t+p[3]+p[4]*x**2/v+p[5]*v+p[6]*x**3