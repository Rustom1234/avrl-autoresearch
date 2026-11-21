def equation(eps, T, params):
    return params[0] + params[1]*eps + params[2]*np.exp(-params[3]*eps) + params[4]*eps**3*(1+params[5]*T+params[6]*T**2) + params[7]*T**params[8]