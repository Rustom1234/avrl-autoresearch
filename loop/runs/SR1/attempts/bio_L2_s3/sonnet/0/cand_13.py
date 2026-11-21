def equation(t,P,params):
    return params[0]*np.exp(-P/params[1])+params[2]+params[3]*t+params[4]*t*P+params[5]*np.sin(params[6]*t+params[7])