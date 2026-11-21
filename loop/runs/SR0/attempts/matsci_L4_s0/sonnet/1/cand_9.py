def equation(eps,T,params):
    return params[0]+params[1]*(T-params[2])*np.log(eps)+params[3]*(T-params[2])**2-params[4]*np.exp(-params[5]*eps**params[6])