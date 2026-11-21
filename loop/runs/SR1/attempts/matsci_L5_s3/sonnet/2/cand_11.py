def equation(eps,T,params):
    return params[0]+params[1]*T+params[2]*T*np.log(T)+params[3]*eps**1.5+params[4]*T**2+params[5]*eps+params[6]*eps*T