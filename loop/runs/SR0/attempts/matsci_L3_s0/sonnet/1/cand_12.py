def equation(eps,T,params):
    return params[0]*np.tanh(eps/params[1])*(1-params[2]*T**2)+params[3]*T**params[4]*eps**params[5]+params[6]*T**2