def equation(eps,T,params):
    return params[0]+params[1]*T+(params[2]+params[3]*T**params[5])*eps**params[4]+params[6]*np.log(eps)