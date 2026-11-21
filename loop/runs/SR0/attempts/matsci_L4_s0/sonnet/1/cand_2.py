def equation(eps,T,params):
    return params[0]+params[1]*np.log(eps)+params[2]*np.exp(-params[3]*eps*1.0)+ params[4]*T**params[5]*eps**params[6]+params[7]*T