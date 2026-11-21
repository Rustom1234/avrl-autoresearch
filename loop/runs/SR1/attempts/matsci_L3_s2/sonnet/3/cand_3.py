def equation(eps,T,params):
    return params[0]+params[1]*np.log(T)+(params[2]*np.exp(-params[3]*eps)+params[4]*eps**params[5])*T**params[6]