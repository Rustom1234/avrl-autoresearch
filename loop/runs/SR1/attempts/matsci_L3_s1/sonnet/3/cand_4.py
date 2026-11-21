def equation(eps,T,params):
    return params[0]+params[1]/T+params[2]*T+params[3]*np.log(eps)+params[4]*eps+params[5]*eps**2+params[6]*eps/T+params[7]*eps*T