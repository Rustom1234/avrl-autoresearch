def equation(eps,T,params):
    return params[0]-params[1]*np.exp(-params[2]*T)+(params[3]*np.exp(-params[4]*eps)+params[5]*eps**params[6])*np.exp(-params[7]*T)+params[8]*np.log(T)