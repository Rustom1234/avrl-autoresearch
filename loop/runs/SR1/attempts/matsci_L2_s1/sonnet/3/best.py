def equation(eps,T,params):
    return params[0]+params[1]*np.exp(-params[2]*T)+eps*(params[4]+params[5]*T)+params[6]*T+params[7]*T**2