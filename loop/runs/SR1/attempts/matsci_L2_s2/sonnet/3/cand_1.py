def equation(eps,T,params):
    return params[0]+params[1]*np.log(eps)+params[2]*np.log(eps)*T+params[3]*T+params[4]*eps**2*T+params[5]*eps+params[6]*eps*T+params[7]*eps**2