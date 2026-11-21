def equation(eps,T,params):
    return params[0]+params[1]*eps+params[2]*T+params[3]/T+params[4]*eps*T+params[5]*eps/T+params[6]*np.log(eps)