def equation(eps, T, params):
    return params[0] + params[1]*np.sqrt(eps**2+params[2]) + params[4]*eps**3*(1+params[5]*T+params[6]*T**2) + params[7]*T**params[8]+params[3]*eps