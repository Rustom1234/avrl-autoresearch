def equation(eps, T, params):
    g = np.exp(-params[0]/T)
    return params[1] + params[2]*T + params[3]*eps**2 + params[4]/eps + g*(params[5] + params[6]*eps + params[7]/eps + params[8]*eps**2)
