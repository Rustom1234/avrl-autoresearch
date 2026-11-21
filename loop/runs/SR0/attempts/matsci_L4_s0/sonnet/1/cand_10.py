def equation(eps,T,params):
    d=T-params[2]
    return params[0]+d*(params[1]*np.log(eps)+params[6])+params[3]*d**2-params[4]*eps**(-params[5])