def equation(eps,T,params):
    return params[0]+params[1]/eps**params[2]+params[3]*eps**params[4]+params[5]*np.exp(params[6]*T)*eps**params[7]