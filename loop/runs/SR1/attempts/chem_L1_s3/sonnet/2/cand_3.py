def equation(t,A,params):
    return -params[0]/A-params[2]*A**params[3]-params[4]+params[5]*np.exp(-params[6]*t)