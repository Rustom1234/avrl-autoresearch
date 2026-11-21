def equation(t,x,v,params):
    p=params
    return p[0]*np.cos(p[1]*t)+p[2]*x+p[3]*x**3+p[4]*v+p[5]/(v+p[6])+p[7]