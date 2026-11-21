def equation(eps,T,params):
    return params[0]/eps**params[1]+params[2]+params[3]*np.log(eps)+params[4]*eps+params[5]*T**params[6]*eps**params[7]