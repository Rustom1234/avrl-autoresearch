def equation(eps,T,params):
    return params[0]+(params[1]+params[6]*T)*np.log(eps)+params[2]*eps**params[3]*T**params[4]+params[5]*eps**params[7]+params[8]*T