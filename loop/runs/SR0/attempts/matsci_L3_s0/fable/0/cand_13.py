def equation(eps, T, params):
    p=params
    return p[0] + p[1]*eps**2 + p[2]*T**2/(eps+p[3]) + p[4]*T + p[5]*T**p[6]*np.exp(p[7]/T) + p[8]*T**2