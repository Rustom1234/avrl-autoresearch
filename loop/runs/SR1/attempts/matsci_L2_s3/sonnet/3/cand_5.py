def equation(eps,T,params):
    return params[0]+params[1]*T+params[2]*eps**2+params[3]*eps+params[4]*T*eps+params[5]*T**2+params[6]*np.sqrt(eps)