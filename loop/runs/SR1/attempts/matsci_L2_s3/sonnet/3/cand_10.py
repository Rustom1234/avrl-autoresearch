def equation(eps,T,params):
    return params[0]+params[1]*np.log(T)+(params[3]+params[4]*T)*eps**params[6]