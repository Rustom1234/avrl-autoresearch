def equation(eps,T,params):
    return params[0]-params[1]/eps+params[2]*np.log(eps)*(T-params[3])*(1+params[4]*T)+params[5]*(T-params[3])**2*params[6]