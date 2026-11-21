def equation(eps,T,params):
    return params[0]+params[1]*T+params[7]*T**2+(params[2]+params[3]*T)*eps**params[4]-params[5]*np.exp(-eps/params[6])*(1+params[8]*T)+params[9]*np.log(eps)