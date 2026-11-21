def equation(eps,T,params):
    return params[0]*eps**params[1]*np.exp(-params[5]*T)+params[2]*T**params[3]+params[4]