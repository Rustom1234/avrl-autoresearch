def equation(eps, T, params):
    return params[0]+params[1]*eps+params[2]*eps**2+params[3]*np.exp(-params[4]*eps)