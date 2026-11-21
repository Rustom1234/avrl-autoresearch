def equation(eps,T,params):
    return params[0]+params[1]*eps**params[2]+params[3]*T+params[4]*T*eps**params[2]+params[5]*np.log(eps)+params[6]*T**2