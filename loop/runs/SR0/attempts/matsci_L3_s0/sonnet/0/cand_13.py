def equation(eps,T,params):
    return params[0]*eps+params[1]*np.exp(-params[2]*(T-params[3])**2)*eps**(params[4]*(T-params[3]))+params[5]*eps*(T-params[3])