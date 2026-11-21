def equation(t,P,params):
    return params[0]-params[1]*P-params[2]*np.log(1+np.exp(params[3]*(P-params[4])))