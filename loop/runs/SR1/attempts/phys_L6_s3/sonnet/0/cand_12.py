def equation(t,x,v,params):
    p=params
    return p[0]*np.cos(p[1]*t)+p[2]*x+p[3]*x**3+p[4]*v+p[5]*np.exp(-p[6]*v)+p[7]