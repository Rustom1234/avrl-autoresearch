def equation(eps, T, params):
    p=params
    return (p[0]*(1 - p[1]*np.exp(-p[2]*T))) * (1 + p[3]*eps*np.exp(-p[4]*eps))