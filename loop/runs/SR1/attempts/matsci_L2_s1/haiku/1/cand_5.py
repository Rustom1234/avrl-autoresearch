def equation(eps, T, params):
    return params[0] + params[1]*T + (params[2] + params[3]*T)*np.exp(-params[4]*eps) + params[5]*eps + params[6]*eps*T