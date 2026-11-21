def equation(eps, T, params):
    p=params
    return p[0]+p[1]*np.abs(eps)**p[2]+p[3]*np.log(np.abs(1+p[4]*T))+p[5]*T