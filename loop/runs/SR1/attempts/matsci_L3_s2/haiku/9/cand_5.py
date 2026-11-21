def equation(eps, T, params):
    return params[0] + params[1]*(1-np.exp(-params[2]*T)) + params[3]*np.exp(-params[4]*eps) + params[5]*eps**2 + params[6]*eps + params[7]*T*np.exp(-params[8]*eps)
