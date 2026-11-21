def equation(t, x, v, params):
    p=params
    return p[0]*np.sin(p[1]*x)+p[2]*(x*v)**p[3]+p[4]/v