def equation(eps, T, params):
    p = params
    return p[0] + p[1]*eps + p[2]/T + p[3]*np.exp(-p[4]/T) + p[5]*T*eps + p[6]*eps/T
