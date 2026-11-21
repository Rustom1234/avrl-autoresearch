def equation(eps,T,params):
    return params[0]+params[1]*T**params[2]*eps**params[3]+params[4]*np.log(eps)+params[5]*eps+params[6]*np.log(eps)**2+params[7]*T