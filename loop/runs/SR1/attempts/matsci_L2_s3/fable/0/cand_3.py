def equation(eps, T, params):
    p=params
    return (p[0] + p[1]*eps**p[2])*(p[3] + p[4]*np.abs(T)**p[5]) + p[6]*np.log(T)+p[7]*T