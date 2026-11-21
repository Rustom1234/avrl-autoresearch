def equation(eps,T,params):
    return params[0]+params[1]*T+params[2]*T**2+params[3]*np.sqrt(T)*eps**2+params[4]*np.log(T)