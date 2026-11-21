def equation(eps,T,params):
    return params[0]*(1-np.exp(-params[1]*eps))*np.exp(-params[2]*T)+params[3]/eps**params[4]+params[5]*eps*T**2+params[6]+params[7]*T