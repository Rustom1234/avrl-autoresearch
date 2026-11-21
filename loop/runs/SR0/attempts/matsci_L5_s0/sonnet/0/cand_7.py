def equation(eps,T,params):
    return params[0]+params[1]*eps+params[2]*np.sin(params[3]*eps+params[4])