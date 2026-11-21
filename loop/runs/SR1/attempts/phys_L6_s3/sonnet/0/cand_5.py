def equation(t,x,v,params):
    p=params
    return p[0]+p[1]*x**p[2]+p[3]*v**p[4]+p[5]*np.sin(p[6]*t+p[7])