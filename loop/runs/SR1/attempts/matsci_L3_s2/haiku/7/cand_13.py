def equation(eps, T, params):
    return params[0] + params[1]*T**params[2] + (params[3]+params[4]*T)*np.exp(-params[5]*eps) + params[6]*eps + params[7]*eps*T
