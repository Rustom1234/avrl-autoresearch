def equation(t,A,params):
    return params[0]*np.exp(-params[1]*t**2-params[4]*A) - params[2]*A**params[3]*t**params[5]