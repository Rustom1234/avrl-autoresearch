def equation(t,P,params):
    return params[0]*P/(params[1]+P) + params[2]*np.cos(params[3]*t+params[4]) + params[5]*t/(P+params[6])+params[7]