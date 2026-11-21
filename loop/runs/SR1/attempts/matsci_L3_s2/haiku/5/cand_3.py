def equation(eps, T, params):
    p=params
    return (p[0] + p[1]*eps**p[2]*np.exp(-p[3]*eps))*(1 - p[4]*np.exp(-p[5]*T))