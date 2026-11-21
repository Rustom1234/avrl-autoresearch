def equation(eps,T,params):
    return params[0]*eps**params[1]+params[2]*T**params[5]*eps**params[3]*(1-np.exp(-params[4]*eps*T))+params[6]*np.log(T)