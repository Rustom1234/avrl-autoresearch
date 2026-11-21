def equation(t,A,params):
    return params[0]*(1+np.exp(-params[2]*params[1]))/(1+np.exp(params[2]*(t-params[1]))) - params[3]*A**params[4]