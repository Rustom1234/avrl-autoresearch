def equation(t,P,params):
    p=params
    return p[0]*np.sqrt(1+np.abs(p[1])*P)*(1+p[2]*np.log(t))+p[3]*t**p[4]*P**2*np.exp(-p[5]*P)+p[6]*P+p[7]*P**2+p[8]+p[9]*np.log(t)