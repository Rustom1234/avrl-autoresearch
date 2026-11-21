def equation(t,A,params):
    return params[0]-params[1]*t-params[2]*A**params[3]+params[4]*np.sin(params[5]*t+params[6])