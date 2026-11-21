def equation(eps,T,params):
    return params[0]+params[1]*T+params[2]*T**2+params[3]*T**3+(params[4]+params[5]*T)*np.sqrt(eps)+(params[6]+params[7]*T)*eps+params[8]*np.log(eps)