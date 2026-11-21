def equation(t,P,params):
    return params[0]*P**params[1]*np.exp(-params[2]*P)+params[3]*P*(1-P/params[4])