def equation(t,P,params):
    p=params
    return p[0]+p[1]*np.log(t)+p[2]*P+p[3]*P**2+p[4]*np.abs(P)**p[5]+p[6]*t/(t+np.abs(p[7])*P)+p[8]*np.sin(p[9]*t)*P