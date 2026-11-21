def equation(t,P,params):
    return params[0]*P*(1-P/params[1])*np.exp(-params[2]*t) - params[3]*t + params[4]*np.sin(params[5]*t+params[6])