def equation(eps, T, params):
    A = params[1] + params[2]*T + params[3]*T**2
    return params[0] + A*eps**params[4]*np.exp(-(params[5]+params[6]*T+params[7]*T**2)*eps) + params[8]*eps + params[9]*eps**2
