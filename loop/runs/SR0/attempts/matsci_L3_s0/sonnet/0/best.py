def equation(eps,T,params):
    return params[0]*eps+params[1]+params[2]*T+params[3]*T**2+(params[4]*T+params[5])*np.log(eps)+params[6]*eps*T