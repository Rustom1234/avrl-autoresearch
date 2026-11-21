def equation(eps,T,params):
    d=T-params[2]
    return params[0]+d*(params[1]*np.log(eps)+params[7])+params[3]*d**2-(params[4]+params[5]*T)*np.exp(-params[6]*eps**params[8])