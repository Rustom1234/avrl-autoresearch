def equation(eps,T,params):
    return params[0]*np.exp(-params[1]*T)+params[2]*eps**params[3]+params[4]*np.log(eps)+params[5]