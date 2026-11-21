def equation(eps,T,params):
    return params[0]-params[1]/eps+params[2]*T+params[3]*T*np.log(eps)+params[4]*np.log(eps)+params[5]*T**2+params[6]*eps+params[7]*T*eps