def equation(t,P,params):
    return params[0]*P/(params[1]+P)*(1+params[2]*np.sin(params[3]*t+params[4]))-params[5]*P+params[6]*t