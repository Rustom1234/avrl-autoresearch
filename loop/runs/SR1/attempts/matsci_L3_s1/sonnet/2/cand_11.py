def equation(eps,T,params):
    return params[0]+params[1]*np.exp(-params[2]*T)*np.log(1+params[3]*eps)*np.exp(-params[4]*eps)+params[5]*T**params[6]+params[7]*eps*T**params[8]