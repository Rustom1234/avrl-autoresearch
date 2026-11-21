def equation(eps, T, params):
    p=params
    return p[0] + p[1]*eps**p[2]*T**p[3] - p[4]*np.exp(p[5]*T)*eps**p[6] + p[7]*T