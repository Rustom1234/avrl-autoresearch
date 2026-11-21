def equation(eps,T,params):
    return params[0]+params[1]*np.log(T)+params[2]*eps+params[3]*eps**params[4]*np.log(T)+params[5]*np.log(eps)*T**params[6]