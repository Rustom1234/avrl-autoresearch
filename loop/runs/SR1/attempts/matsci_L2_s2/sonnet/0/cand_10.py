def equation(eps,T,params):
    return params[0]+params[1]*np.log(eps)+params[2]/eps+params[4]*np.exp(params[3]*eps*T)