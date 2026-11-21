def equation(eps,T,params):
    return params[0]+params[1]*eps**params[2]*(1+params[3]*T)+params[4]*T+params[5]*np.exp(-params[6]*T)