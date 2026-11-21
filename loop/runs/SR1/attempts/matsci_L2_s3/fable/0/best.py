def equation(eps, T, params):
    p=params
    return p[0]*T/(T+p[1]) + p[2]*T**2 + p[3]*np.sqrt(T)*eps**2