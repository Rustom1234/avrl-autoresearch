def equation(eps,T,params):
    return params[0]+params[1]*np.log(eps+params[5])+params[3]*T+params[4]*T*np.log(eps+params[5])+params[2]*T**2