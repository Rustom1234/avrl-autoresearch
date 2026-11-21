def equation(eps, T, params):
    p=params
    return p[0] + p[1]*np.abs(T)**p[2] + p[3]*eps**p[4]*np.abs(T)**p[5] + p[6]*eps**p[7]