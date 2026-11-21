def equation(eps,T,params):
    return params[0]+params[1]*np.exp(-params[2]*T)+ (params[3]+params[4]*T)*(1-np.exp(-params[5]*eps))*np.exp(-params[6]*eps)+params[7]*eps