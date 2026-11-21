def equation(t,A,params):
    return params[0]*np.cos(params[1]*t) - params[3]*A**params[4]