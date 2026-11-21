def equation(t,P,params):
    return P*(params[0]-params[1]*P)-params[2]*np.exp(params[3]*t)/(P+params[4])+params[5]*np.sin(params[6]*t+params[7])