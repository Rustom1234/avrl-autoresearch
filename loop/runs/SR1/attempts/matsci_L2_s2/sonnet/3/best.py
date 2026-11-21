def equation(eps,T,params):
    return params[0]+params[1]*np.log(eps+params[2])+params[3]*T**2*eps**3+params[4]*eps**params[5]+params[6]*np.exp(-params[7]*eps)