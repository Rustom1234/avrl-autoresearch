def equation(eps,T,params):
    return params[0]+params[1]*T+params[5]*T**2+(params[2]+params[3]*T)*np.sqrt(eps)