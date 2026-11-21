def equation(eps,T,params):
    return params[0]*eps+params[1]+params[4]*T+(params[2]*T+params[3])*np.log(eps)