def equation(eps,T,params):
    return params[0]+params[1]*T+params[2]*np.log(eps*T**params[3]+params[4])+params[5]*eps*T**params[6]+params[7]*eps