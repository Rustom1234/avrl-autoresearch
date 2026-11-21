def equation(t,P,params):
    return params[0]*P**params[1]*np.exp(-P/params[2])+params[3]+t*(params[4]+params[5]*P)