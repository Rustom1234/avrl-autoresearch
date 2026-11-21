def equation(eps,T,params):
    return params[0]+params[1]*T+params[2]*T**2+params[3]*eps**1.5+params[4]*np.exp(-T/params[5])