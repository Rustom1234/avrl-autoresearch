def equation(eps,T,params):
    return params[0]+params[1]*T**params[2]*eps**params[3]+params[4]*eps**params[5]+params[6]*np.exp(-eps/params[7])+params[8]*np.log(eps)