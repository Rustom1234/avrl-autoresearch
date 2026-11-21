def equation(eps, T, params):
    p=params
    return (p[0] - p[1]*np.exp(-p[2]*T)) * (1 + p[3]*np.exp(-p[4]*eps*T) + p[5]*eps) + p[6]*np.exp(-p[7]*eps)