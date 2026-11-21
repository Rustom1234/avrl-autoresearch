def equation(t, x, v, params):
    p=params
    return p[0]*np.log(1+p[1]*t)+p[2]*t/x+p[3]*x**3