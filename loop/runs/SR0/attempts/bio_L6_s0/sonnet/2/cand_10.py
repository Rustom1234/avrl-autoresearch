def equation(t,P,params):
    return P*(params[0]+params[1]*P+params[2]*t)+params[3]*t*(1+params[4]*np.sin(params[5]*t+params[6]))