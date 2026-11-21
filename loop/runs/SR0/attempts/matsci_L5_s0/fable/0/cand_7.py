def equation(eps, T, params):
    return params[0]+params[1]*eps+params[2]*np.sqrt(params[3]**2+eps**2)+params[4]*eps**2