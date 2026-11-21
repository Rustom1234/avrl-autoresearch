def equation(eps, T, params):
    return params[0] + params[1]*T + params[2]/T + params[3]*eps**2 + params[4]/eps + params[5]*eps + params[6]*eps/T + params[7]*np.exp(-params[8]*T) + params[9]/(eps*T)
