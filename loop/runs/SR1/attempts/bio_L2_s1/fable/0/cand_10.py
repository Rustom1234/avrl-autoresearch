def equation(t, P, params):
    p=params
    return p[0]*np.sqrt(P)*t+p[1]+p[2]*P+p[3]*P**2+p[4]*P**p[5]+p[6]*P**2/(p[7]**2+P**2)