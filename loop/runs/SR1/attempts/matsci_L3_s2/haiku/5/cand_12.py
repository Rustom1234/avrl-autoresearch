def equation(eps, T, params):
    p=params
    return (p[0] - p[1]*T**(-p[2])) * (1 + p[3]*np.exp(-p[4]*eps) + p[5]*eps)