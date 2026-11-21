def equation(t, P, params):
    p=params
    return p[0]+p[1]*P+p[2]*P**2+p[3]*np.abs(P)**p[4]+p[5]*np.abs(t)**p[6]+p[7]*P*t+p[8]*np.log(t)+p[9]*np.log(P)