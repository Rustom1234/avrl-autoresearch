def equation(eps, T, params):
    return params[0]*np.exp(-params[1]*eps) + params[2] + params[3]*eps + params[4]*eps**2 + params[5]*eps**3 + params[6]*np.exp(-params[7]*eps**2)
