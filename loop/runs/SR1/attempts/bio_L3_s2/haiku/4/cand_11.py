def equation(t,P,params):
    return params[0]*(t*P)**params[1]+params[2]*np.exp(-t)+params[3]*P