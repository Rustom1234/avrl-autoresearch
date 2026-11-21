def equation(t,P,params):
    return params[0]*P*(1-P/params[1]) - params[2]*t**params[3] + params[4]*np.log(P)+params[5]*np.sin(params[6]*t)