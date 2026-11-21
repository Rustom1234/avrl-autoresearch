def equation(t,P,params):
    return params[0]*P*(1-P/params[1])-params[2]*P/(params[3]+P)+params[4]*np.sin(params[5]*t)