def equation(eps,T,params):
    return params[0]+params[1]*eps+params[2]*np.tanh(params[3]*eps)+params[4]*T*eps+params[5]*eps**2+params[6]*T