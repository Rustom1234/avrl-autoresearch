def equation(eps, T, params):
    g = params[0] + params[1]*(1-np.exp(-params[2]*T))
    h = 1 + params[3]*np.exp(-params[4]*eps) + params[5]*eps + params[6]*eps**2
    return g*h
