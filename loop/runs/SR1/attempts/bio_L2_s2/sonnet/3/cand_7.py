def equation(t,P,params):
    return params[0]*P/(params[1]+P)+t*(params[2]+params[3]*P)+params[4]*np.sin(params[5]*t)*(params[6]+P)