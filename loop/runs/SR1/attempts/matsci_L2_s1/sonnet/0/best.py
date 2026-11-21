def equation(eps,T,params):
    return params[0]+params[1]*T+params[2]*np.exp(-params[5]*T)+eps*(params[3]+params[4]*T)+params[6]*T**2