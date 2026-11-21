def equation(eps,T,params):
    return params[0]*(1-np.exp(-params[1]*T))+(params[2]*eps**(-params[3])+params[4]*eps**params[5])*(1-np.exp(-params[6]*T))+params[7]*np.log(eps)*np.exp(-params[8]*T)