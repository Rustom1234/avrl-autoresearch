def equation(eps, T, params):
    return -(params[0]*eps**params[1] + params[2]*eps**params[3])*np.exp(params[4]*T + params[5]*T*eps)
