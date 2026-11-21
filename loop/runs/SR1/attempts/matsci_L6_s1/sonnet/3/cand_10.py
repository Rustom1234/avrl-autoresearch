def equation(eps,T,params):
    return params[0]+params[1]*np.log(T)+params[2]*eps**params[3]*T**params[4]+params[5]*np.log(T)*eps**params[6]