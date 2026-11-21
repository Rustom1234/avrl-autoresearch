def equation(eps, T, params):
    return params[0]*np.sin(params[1]*eps+params[2]*T+params[3])*np.exp(-params[4]*eps) + params[5]*np.cos(params[6]*eps+params[7]*T) + params[8]*T + params[9]