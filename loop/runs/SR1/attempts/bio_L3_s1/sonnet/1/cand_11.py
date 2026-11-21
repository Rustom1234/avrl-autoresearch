def equation(t,P,params):
    return params[0]/P + params[1]*P + params[5] + params[2]*t**params[6]*np.sin(params[3]*P+params[4])