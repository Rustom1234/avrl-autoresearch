def equation(t,P,params):
    return params[0]*P*(1-P/params[1])-params[2]*np.exp(-P/params[3])+params[4]*np.sin(params[5]*t+params[6])