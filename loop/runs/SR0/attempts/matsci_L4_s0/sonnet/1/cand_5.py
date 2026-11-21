def equation(eps,T,params):
    return params[0]*T**params[1]*eps**params[2]+params[3]+params[4]*np.log(eps)+params[5]*eps**(-params[6])