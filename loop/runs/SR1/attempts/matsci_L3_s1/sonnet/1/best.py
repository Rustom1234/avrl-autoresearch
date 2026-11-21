def equation(eps,T,params):
    return params[0]+params[1]*T+params[2]*T**2+(params[3]+params[4]*T+params[7]*T**2)*eps**params[5]*np.exp(-params[6]*eps)+params[8]*np.log(eps)