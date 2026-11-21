def equation(eps,T,params):
    g=np.log1p(params[0]*eps)*np.exp(-params[1]*eps)
    h=params[2]/T+params[3]*T+params[4]
    return params[5]+h*(1+params[6]*g)+params[7]*g