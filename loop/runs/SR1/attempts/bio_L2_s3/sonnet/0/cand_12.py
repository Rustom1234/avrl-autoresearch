def equation(t,P,params):
    return params[0]*np.exp(-P/params[1])+params[2]*t*(1-np.exp(-P/params[3]))+params[4]+params[5]*t