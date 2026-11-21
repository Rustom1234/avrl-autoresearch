def equation(eps,T,params):
    return params[0]+params[1]*eps*(1+params[2]*T)+params[3]*T+params[4]*np.exp(-params[5]*T)+params[6]*eps*np.exp(-params[7]*T)