def equation(eps, T, params):
    return params[0]+params[1]*np.log(eps)+params[3]*eps+params[2]*eps**2+(params[4]+params[5]*eps+params[6]*np.log(eps))*np.exp(-params[7]*T)