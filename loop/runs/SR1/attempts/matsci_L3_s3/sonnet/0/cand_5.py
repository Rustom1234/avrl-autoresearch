def equation(eps,T,params):
    return params[0]+params[1]*T+params[2]*np.log(eps+params[5])*np.exp(-params[4]*T)