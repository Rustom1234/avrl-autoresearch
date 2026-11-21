def equation(t,P,params):
    return params[0]*np.exp(-P/params[1])+params[2]*t*P/(P+params[3])+params[4]