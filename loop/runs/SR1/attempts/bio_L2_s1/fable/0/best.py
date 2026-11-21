def equation(t, P, params):
    p=params
    return p[0]*np.sqrt(P)*t+p[1]+p[2]*np.exp(-p[3]*P)+p[4]*P+p[5]*P**p[6]+p[7]*P**p[8]