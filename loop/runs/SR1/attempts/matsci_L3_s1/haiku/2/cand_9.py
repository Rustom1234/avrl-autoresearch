def equation(eps, T, params):
    A = params[0] + params[1]*T + params[2]*T**2
    return A*eps**params[3]*np.exp(-(params[4]+params[5]*T)*eps) + params[6] + params[7]*T + params[8]*T**2
